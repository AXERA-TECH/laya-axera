"""Laya moves the AI paddle; the option wording is the bricks game's, verbatim.

Same labels, state, instructions and tier texts as laya_axera.bricks.policy, so
the inputs are exactly those probed there. Every possible tier assignment (12,
of which 7 can occur) was run on the NPU and the model picks the planner's
action in all of them.
"""

import math
import time

from ..bricks.policy import AWAY, BEST, INSTRUCTIONS, STATE, WALL
from .planner import ACTIONS, plan

TEXT = {"best": BEST, "away": AWAY, "wall": WALL}


def _number(v, name):
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number")
    return float(v)


def _validate(court, ball, ai):
    w = _number(court.get("width"), "court.width")
    _number(court.get("ai_y"), "court.ai_y")
    _number(court.get("player_y"), "court.player_y")
    for key in ("x", "y", "vx", "vy"):
        _number(ball.get(key), f"ball.{key}")
    for key in ("x", "w", "step"):
        _number(ai.get(key), f"ai.{key}")
    if not 2 <= w <= 100 or not 0 < ai["w"] < w or ai["step"] < 0:
        raise ValueError("court and paddle sizes are out of range")
    if not court["ai_y"] < court["player_y"]:
        raise ValueError("court.ai_y must be above court.player_y")


class LayaPaddlePolicy:
    def __init__(self, agent):
        self.agent = agent

    def decide(self, court, ball, ai):
        _validate(court, ball, ai)
        facts = plan(court, ball, ai)
        criteria = {a: TEXT[facts["tiers"][a]] for a in ACTIONS}
        started = time.perf_counter()
        output = self.agent.predict(
            STATE, {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}
        )
        inference_ms = (time.perf_counter() - started) * 1000
        probabilities = output["answers"]["action"]["probabilities"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError("Model returned an invalid probability; no action taken")
        return {
            "action": max(probabilities, key=probabilities.__getitem__),
            "probabilities": probabilities,
            "planner_best": facts["best"],
            "target": round(facts["target"], 3),
            "error": round(facts["error"], 3),
            "inference_ms": round(inference_ms, 3),
        }
