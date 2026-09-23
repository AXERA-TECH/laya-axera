"""One Laya question per decision step: move the paddle left, right, or hold.

The wording is probe-selected on the multilingual checkpoint and is the same
three-tier scheme the Flappy Bird policy uses (see NOTICE): neutral action
labels, a constant minimal state, and the planner's verdict carried entirely by
the option descriptions. Probed over every rotation of which action is best,
the three-way choice picks the intended action with 0.62-0.78 probability.
"""

import math
import time
from dataclasses import asdict, dataclass

from .game import ACTIONS

BEST = "Safe. Best route to the ball."
AWAY = "Unsafe. Away from the ball."
WALL = "Blocked. Wall."
STATE = "Track the ball."
INSTRUCTIONS = "Choose the best safe action."


@dataclass
class BreakoutDecision:
    probabilities: dict
    proposed: str
    executed: str
    intervened: bool
    target: float
    error: float
    planner_best: str
    inference_ms: float
    decision_ms: float
    input_tokens: int

    def to_dict(self):
        return asdict(self)


class LayaBreakoutPolicy:
    def __init__(self, agent, *, guarded=True):
        self.agent = agent
        self.guarded = guarded

    def decide(self, game):
        started = time.perf_counter()
        plan = game.plan()
        criteria = {}
        for action in ACTIONS:
            info = plan["options"][action]
            if info["blocked"]:
                criteria[action] = WALL
            elif action == plan["best"]:
                criteria[action] = BEST
            else:
                criteria[action] = AWAY
        questions = {
            "action": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            }
        }
        inference_start = time.perf_counter()
        output = self.agent.predict(STATE, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        probabilities = output["answers"]["action"]["probabilities"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError("Model returned an invalid probability; no action executed")
        proposed = max(probabilities, key=probabilities.__getitem__)
        executed = proposed
        if self.guarded and plan["options"][proposed]["fatal"]:
            rescues = [a for a in ACTIONS if not plan["options"][a]["fatal"]]
            if rescues:
                executed = min(rescues, key=lambda a: plan["options"][a]["error"])
        return BreakoutDecision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            intervened=proposed != executed,
            target=round(plan["target"], 3),
            error=round(plan["error"], 3),
            planner_best=plan["best"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
        )
