"""Where the ball will reach the AI's paddle line, and what each move does about it.

The table is seen from above: the AI's paddle sits near y = 0, the player's near
y = length. The ball reflects off the side rails. While it travels away from the
AI, the planner assumes the player returns it straight, so the AI can already
drift toward where it will come back.
"""

import math

ACTIONS = ("left", "right", "hold")
MAX_SIM_S = 6.0
SIM_DT = 0.005


def predict_x(court, ball):
    """x where the ball next crosses the AI's line, and seconds until then."""
    width, ai_y, player_y = court["width"], court["ai_y"], court["player_y"]
    r = ball.get("r", 0.25)
    x, y, vx, vy = ball["x"], ball["y"], ball["vx"], ball["vy"]
    t = 0.0
    while t < MAX_SIM_S:
        x += vx * SIM_DT
        y += vy * SIM_DT
        t += SIM_DT
        if x < r:
            x, vx = 2 * r - x, abs(vx)
        elif x > width - r:
            x, vx = 2 * (width - r) - x, -abs(vx)
        if vy > 0 and y >= player_y:
            y, vy = 2 * player_y - y, -abs(vy)
        if vy < 0 and y <= ai_y:
            return x, t
    return x, t


DEADBAND = 0.3


def moved(x, direction, step, target, lo, hi):
    """Where the paddle ends up. A move toward the predicted landing point stops on
    it, so a late next decision (network delay) makes the AI slow, not wild. The page
    applies exactly this rule."""
    if direction == "hold":
        return x
    sign = -1 if direction == "left" else 1
    end = x + sign * step
    if (target - x) * sign > 0:
        end = min(end, target) if sign > 0 else max(end, target)
    return max(lo, min(hi, end))


def plan(court, ball, ai):
    """Best action and a tier per action for the AI paddle."""
    width = court["width"]
    half = ai["w"] / 2
    target, eta = predict_x(court, ball)
    x = ai["x"]
    after = {a: moved(x, a, ai["step"], target, half, width - half) for a in ACTIONS}
    blocked = {a: a != "hold" and math.isclose(after[a], x) for a in ACTIONS}
    error = {a: abs(after[a] - target) for a in ACTIONS}
    if abs(x - target) < DEADBAND:
        error["hold"] = 0.0
    candidates = [a for a in ACTIONS if not blocked[a]]
    best = min(candidates, key=lambda a: (error[a], a != "hold"))
    tiers = {a: "best" if a == best else "wall" if blocked[a] else "away" for a in ACTIONS}
    return {"best": best, "tiers": tiers, "target": target, "eta": eta, "error": abs(x - target)}
