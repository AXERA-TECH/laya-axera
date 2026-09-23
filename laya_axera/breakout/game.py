"""Deterministic Breakout rules in cell units, with a landing-point planner.

The y axis grows downward (canvas convention). One decision step advances
FRAMES_PER_STEP physics frames; the chosen action shifts the paddle once at the
start of the step.
"""

import math
import random

WIDTH, HEIGHT = 20, 14
FRAMES_PER_STEP = 3
BALL_R = 0.3
BALL_SPEED = 0.42
PADDLE_W = 3.2
PADDLE_Y = HEIGHT - 1.0
PADDLE_STEP = 1.1
BRICK_ROWS = (2, 3, 4, 5)
BRICK_W = 2
LIVES = 3
ACTIONS = ("left", "right", "hold")


class BreakoutGame:
    def __init__(self, seed=7):
        self.seed = seed
        self.rng = random.Random(seed)
        self.bricks = {(r, c) for r in BRICK_ROWS for c in range(WIDTH // BRICK_W)}
        self.paddle_x = WIDTH / 2
        self.lives = LIVES
        self.score = self.steps = 0
        self.alive = True
        self.won = False
        self.end_reason = None
        self._launch()

    def _launch(self):
        self.ball_x = self.paddle_x
        self.ball_y = PADDLE_Y - 1.0
        angle = self.rng.uniform(-0.6, 0.6)
        self.ball_vx = BALL_SPEED * math.sin(angle)
        self.ball_vy = -BALL_SPEED * math.cos(angle)

    @staticmethod
    def _brick_at(x, y):
        row = int(y)
        if row not in BRICK_ROWS or not 0 <= x < WIDTH:
            return None
        return (row, int(x) // BRICK_W)

    def _advance(self, ball, bricks, paddle_x, collide_paddle=True):
        """One physics frame on a (x, y, vx, vy) ball; returns the ball and an outcome."""
        x, y, vx, vy = ball
        x += vx
        y += vy
        if x < BALL_R:
            x, vx = BALL_R, abs(vx)
        elif x > WIDTH - BALL_R:
            x, vx = WIDTH - BALL_R, -abs(vx)
        if y < BALL_R:
            y, vy = BALL_R, abs(vy)
        hit = self._brick_at(x, y)
        outcome = None
        if hit is not None and hit in bricks:
            bricks.discard(hit)
            vy = -vy
            y += vy * 0.5
            outcome = "brick"
        elif collide_paddle and vy > 0 and PADDLE_Y <= y + BALL_R <= PADDLE_Y + 0.6:
            if abs(x - paddle_x) <= PADDLE_W / 2 + BALL_R:
                # Hitting off-center angles the bounce; renormalize to a constant speed
                # so the rally keeps a predictable pace.
                vx += 0.22 * (x - paddle_x) / (PADDLE_W / 2)
                vy = -abs(vy)
                speed = math.hypot(vx, vy)
                vx, vy = vx * BALL_SPEED / speed, vy * BALL_SPEED / speed
                y = PADDLE_Y - BALL_R
                outcome = "paddle"
            elif y - BALL_R > HEIGHT:
                outcome = "lost"
        elif y - BALL_R > HEIGHT:
            outcome = "lost"
        return (x, y, vx, vy), outcome

    def predict_landing(self, max_frames=400):
        """Where the ball will cross the paddle row if nothing intercepts it."""
        ball = (self.ball_x, self.ball_y, self.ball_vx, self.ball_vy)
        bricks = set(self.bricks)
        for frames in range(1, max_frames + 1):
            ball, _ = self._advance(ball, bricks, self.paddle_x, collide_paddle=False)
            if ball[3] > 0 and ball[1] + BALL_R >= PADDLE_Y:
                return ball[0], frames
        return ball[0], max_frames

    def paddle_after(self, action):
        dx = {"left": -PADDLE_STEP, "right": PADDLE_STEP, "hold": 0.0}[action]
        half = PADDLE_W / 2
        return max(half, min(WIDTH - half, self.paddle_x + dx))

    def plan(self):
        """Facts for the prompt: landing point, per-action error, and reachability."""
        target, frames_left = self.predict_landing()
        steps_left = max(1, frames_left / FRAMES_PER_STEP)
        options = {}
        for action in ACTIONS:
            after = self.paddle_after(action)
            blocked = action != "hold" and abs(after - self.paddle_x) < 1e-9
            error = abs(after - target)
            reach = PADDLE_STEP * (steps_left - 1) + PADDLE_W / 2 + BALL_R
            options[action] = {
                "blocked": blocked,
                "error": error,
                "fatal": error > reach,
            }
        best = min(
            (a for a in ACTIONS if not options[a]["blocked"]),
            key=lambda a: options[a]["error"],
            default="hold",
        )
        return {
            "target": target,
            "frames_left": frames_left,
            "error": abs(self.paddle_x - target),
            "options": options,
            "best": best,
        }

    def step(self, action):
        if not self.alive or self.won:
            raise RuntimeError("Cannot step a finished game")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        self.paddle_x = self.paddle_after(action)
        ball = (self.ball_x, self.ball_y, self.ball_vx, self.ball_vy)
        for _ in range(FRAMES_PER_STEP):
            ball, outcome = self._advance(ball, self.bricks, self.paddle_x)
            if outcome == "brick":
                self.score += 10
                if not self.bricks:
                    self.won = True
                    self.end_reason = "cleared"
                    break
            elif outcome == "lost":
                self.lives -= 1
                if self.lives <= 0:
                    self.alive = False
                    self.end_reason = "missed"
                else:
                    self.ball_x, self.ball_y = ball[0], ball[1]
                    self._launch()
                    ball = (self.ball_x, self.ball_y, self.ball_vx, self.ball_vy)
                break
        self.ball_x, self.ball_y, self.ball_vx, self.ball_vy = ball
        self.steps += 1
        return self.alive and not self.won

    def snapshot(self):
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "seed": self.seed,
            "ball": [round(self.ball_x, 3), round(self.ball_y, 3)],
            "ball_r": BALL_R,
            "paddle_x": round(self.paddle_x, 3),
            "paddle_w": PADDLE_W,
            "paddle_y": PADDLE_Y,
            "brick_w": BRICK_W,
            "bricks": sorted([r, c] for r, c in self.bricks),
            "bricks_left": len(self.bricks),
            "score": self.score,
            "lives": self.lives,
            "steps": self.steps,
            "alive": self.alive,
            "won": self.won,
            "end_reason": self.end_reason,
        }
