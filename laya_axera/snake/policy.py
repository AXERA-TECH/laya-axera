"""Real Laya predictions on the NPU, with an optional deterministic safety shield.

Adapted from laya-mlx (see NOTICE). The move question is probe-selected on the
multilingual checkpoint: labels up / down / left / east ("right" also means
"correct" and pulled probability toward itself; "north" did the same in this
context), a constant state, and four fixed option tiers. That makes the move
question's whole input space 4 best positions x 3^3 tier assignments = 108
cases; all 108 were probed and the model picks the planner's move in every one,
with a smallest winning margin of 0.161. Two noul questions (is a safe route
available, is the food reachable) feed the page's gauges.
"""

import math
import time
from dataclasses import asdict, dataclass

from .game import DIRECTIONS

LABEL = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "east"}
DIRECTION = {v: k for k, v in LABEL.items()}
MOVE_STATE = "Stay alive and eat."
MOVE_INSTRUCTIONS = "Choose the best safe action."
TEXT = {
    "best": "Safe. Best move.",
    "wrong": "Wrong way.",
    "trap": "Unsafe. Traps the snake.",
    "wall": "Blocked. Wall.",
}


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(self, agent, *, guarded=True):
        self.agent = agent
        self.guarded = guarded

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        preferred = game.preferred(moves)
        if preferred is None:
            raise RuntimeError("No legal move left")
        tiers = {}
        for m in moves:
            if m.direction == preferred:
                tiers[m.direction] = "best"
            elif not m.legal:
                tiers[m.direction] = "wall"
            elif not m.safe:
                tiers[m.direction] = "trap"
            else:
                tiers[m.direction] = "wrong"
        criteria = {LABEL[d]: TEXT[tiers[d]] for d in DIRECTIONS}
        reachable, _ = game.food_reachability()
        facts = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        inference_start = time.perf_counter()
        move = self.agent.predict(
            MOVE_STATE,
            {"move": {"type": "choice", "instructions": MOVE_INSTRUCTIONS, "criteria": criteria}},
        )
        gauges = self.agent.predict(
            facts,
            {
                "risk": {"type": "noul", "instructions": "Is a safe route available?"},
                "food": {"type": "noul", "instructions": "Is food reachable through empty cells?"},
            },
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000
        probabilities = {DIRECTION[k]: v for k, v in move["answers"]["move"]["probabilities"].items()}
        risk, food = gauges["answers"]["risk"]["noul"], gauges["answers"]["food"]["noul"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [*probabilities.values(), risk, food]):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and allowed and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - risk,
            food_reachable=food,
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=move["usage"]["input_tokens"] + gauges["usage"]["input_tokens"],
            output_tokens=0,
            safe_count=len(safe),
            planner_best=preferred,
        )
