"""One Laya question per decision step: flap or hold, with planner-described options.

Same structure as the Snake policy (see NOTICE): the deterministic game supplies
facts, the model chooses, and the optional shield corrects only fatal proposals.
"""

import math
import time
from dataclasses import asdict, dataclass


@dataclass
class BirdDecision:
    probabilities: dict
    proposed: str
    executed: str
    intervened: bool
    gap_offset: float
    pipe_distance: float
    autopilot_best: str
    inference_ms: float
    decision_ms: float
    input_tokens: int

    def to_dict(self):
        return asdict(self)


class LayaBirdPolicy:
    def __init__(self, agent, *, guarded=True):
        self.agent = agent
        self.guarded = guarded

    @staticmethod
    def _describe(safe, toward_gap):
        # Wording is probe-selected on the multilingual checkpoint: neutral labels
        # ("flap" itself attracts probability mass), a constant minimal state
        # (directional words in the state interfere with the labels), and the
        # planner's verdict carried entirely by these three tiers.
        if not safe:
            return "Blocked. Collision."
        if toward_gap:
            return "Safe. Best route to the gap."
        return "Unsafe. Away from the gap."

    def decide(self, game):
        started = time.perf_counter()
        plan = game.plan()
        questions = {
            "action": {
                "type": "choice",
                "instructions": "Choose the best safe action.",
                "criteria": {
                    "up": self._describe(plan["safe_flap"], plan["offset"] > plan["deadband"]),
                    "down": self._describe(plan["safe_hold"], plan["offset"] <= plan["deadband"]),
                },
            }
        }
        state = "Fly through the gap."
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        probabilities = output["answers"]["action"]["probabilities"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError("Model returned an invalid probability; no action executed")
        proposed = max(probabilities, key=probabilities.__getitem__)
        executed = proposed
        if self.guarded and not plan["safe_flap" if proposed == "up" else "safe_hold"]:
            other = "down" if proposed == "up" else "up"
            if plan["safe_flap" if other == "up" else "safe_hold"]:
                executed = other
        return BirdDecision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            intervened=proposed != executed,
            gap_offset=round(plan["offset"], 3),
            pipe_distance=round(plan["distance"], 3),
            autopilot_best=plan["autopilot"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
        )
