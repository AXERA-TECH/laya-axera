"""Laya picks the AI tank's direction from the planner's verdicts.

Wording is probe-selected on the multilingual checkpoint. Compass labels
replace up/down/left/right: "right" also means "correct" and pulled
probability toward itself (97/108 with it, 108/108 without), and a "fire"
label was dropped for the same reason - shooting is a mode of a direction.
The state and instructions are constant and every option text comes from four
fixed tiers, so the whole input space is 4 best positions x 3^3 tier
assignments = 108 cases; all 108 were probed and the model picks the planner's
best in every one.
"""

import math
import time

from .planner import BRICK, DIRS, EMPTY, ORDER, STEEL, plan

STATE = "Win the tank battle."
INSTRUCTIONS = "Choose the best safe action."
TEXT = {
    "best": "Safe. Best move.",
    "wrong": "Wrong way.",
    "risk": "Unsafe. Into the line of fire.",
    "wall": "Blocked. Wall.",
}
MAX_SIDE = 32


def _validate(grid, ai, player, bullets):
    if not isinstance(grid, list) or not grid or not all(isinstance(r, list) for r in grid):
        raise ValueError("grid must be a nonempty list of rows")
    width = len(grid[0])
    if len(grid) > MAX_SIDE or width > MAX_SIDE or any(len(r) != width for r in grid):
        raise ValueError(f"grid must be rectangular and at most {MAX_SIDE} tiles a side")
    if any(v not in (EMPTY, BRICK, STEEL) for r in grid for v in r):
        raise ValueError("grid tiles must be 0 (empty), 1 (brick) or 2 (steel)")
    for who, tank in (("ai", ai), ("player", player)):
        x, y = tank.get("x"), tank.get("y")
        if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < width and 0 <= y < len(grid)):
            raise ValueError(f"{who} position is outside the grid")
    if ai.get("dir") not in DIRS:
        raise ValueError("ai.dir must be north, south, west or east")
    if not isinstance(bullets, list) or len(bullets) > 16:
        raise ValueError("bullets must be a list of at most 16 entries")


class LayaTankPolicy:
    def __init__(self, agent):
        self.agent = agent

    def decide(self, grid, ai, player, bullets):
        _validate(grid, ai, player, bullets)
        facts = plan(grid, ai, player, bullets)
        criteria = {name: TEXT[facts["tiers"][name]] for name in ORDER}
        started = time.perf_counter()
        output = self.agent.predict(
            STATE,
            {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
        )
        inference_ms = (time.perf_counter() - started) * 1000
        probabilities = output["answers"]["action"]["probabilities"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError("Model returned an invalid probability; no action taken")
        pick = max(probabilities, key=probabilities.__getitem__)
        return {
            "action": pick,
            "mode": facts["modes"][pick],
            "modes": facts["modes"],
            "probabilities": probabilities,
            "planner_best": facts["best"],
            "reason": facts["reason"],
            "criteria": criteria,
            "inference_ms": round(inference_ms, 3),
        }
