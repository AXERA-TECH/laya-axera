"""Deterministic Flappy Bird rules in cell units, with a 2-ply safety planner.

The y axis grows downward (canvas convention). One decision step advances
FRAMES_PER_STEP physics frames; a flap sets the vertical velocity once at the
start of the step.
"""

import random

# Finer frames than one decision needs: the trajectory per decision is unchanged
# (velocities scale by 1/k, gravity by 1/k^2), but each step now carries a
# sub-frame trail the client plays back instead of a single jump.
FRAMES_PER_STEP = 10
GRAVITY = 0.0088  # cells per frame^2
FLAP_VY = -0.168  # cells per frame, set (not added) on flap
MAX_VY = 0.24
SCROLL = 0.044  # cells per frame
BIRD_X = 6.0
BIRD_R = 0.32
PIPE_W = 1.6


class FlappyGame:
    def __init__(self, width=24, height=16, seed=7, gap=4.4, spacing=9.0):
        if not 3.0 <= gap <= height - 4:
            raise ValueError("gap must leave room for pipes")
        self.width, self.height = width, height
        self.gap, self.spacing = float(gap), float(spacing)
        self.seed = seed
        self.rng = random.Random(seed)
        self.y = height / 2.0
        self.vy = 0.0
        self.pipes = []  # {"x": center, "gap_y": gap center, "passed": bool}
        x = width + 2.0
        while x < width + 3 * spacing:
            self.pipes.append(self._new_pipe(x))
            x += spacing
        self.score = self.steps = self.frames = 0
        self.trail = []
        self.alive = True
        self.death_reason = None

    def _new_pipe(self, x):
        margin = self.gap / 2 + 1.2
        return {
            "x": x,
            "gap_y": self.rng.uniform(margin, self.height - margin),
            "passed": False,
        }

    def next_pipe(self):
        """The nearest pipe whose trailing edge is still ahead of the bird."""
        candidates = [p for p in self.pipes if p["x"] + PIPE_W / 2 + BIRD_R > BIRD_X]
        return min(candidates, key=lambda p: p["x"])

    def _advance_frame(self):
        self.frames += 1
        self.vy = min(self.vy + GRAVITY, MAX_VY)
        self.y += self.vy
        if self.y < BIRD_R:  # ceiling is a clamp, not a death
            self.y, self.vy = BIRD_R, 0.0
        if self.y > self.height - BIRD_R:
            self.alive, self.death_reason = False, "floor"
            return
        for pipe in self.pipes:
            pipe["x"] -= SCROLL
            if not pipe["passed"] and pipe["x"] + PIPE_W / 2 < BIRD_X - BIRD_R:
                pipe["passed"] = True
                self.score += 1
            if abs(pipe["x"] - BIRD_X) < PIPE_W / 2 + BIRD_R:
                half = self.gap / 2
                if self.y - BIRD_R < pipe["gap_y"] - half or self.y + BIRD_R > pipe["gap_y"] + half:
                    self.alive, self.death_reason = False, "pipe"
                    return
        if self.pipes and self.pipes[0]["x"] < -2.0:
            self.pipes.pop(0)
            self.pipes.append(self._new_pipe(self.pipes[-1]["x"] + self.spacing))

    def step(self, flap: bool):
        """Advance one decision step; `trail` holds each frame for client playback.

        A trail entry is [bird_y, pipe_shift], where pipe_shift is added to the
        final pipe positions to place them at that frame.
        """
        if not self.alive:
            raise RuntimeError("Cannot step a finished game")
        if flap:
            self.vy = FLAP_VY
        frames = []
        for _ in range(FRAMES_PER_STEP):
            self._advance_frame()
            frames.append(round(self.y, 3))
            if not self.alive:
                break
        total = SCROLL * len(frames)
        self.trail = [
            [y, round(total - SCROLL * (i + 1), 3)] for i, y in enumerate(frames)
        ]
        self.steps += 1
        return self.alive

    def _clone(self):
        twin = object.__new__(FlappyGame)
        twin.width, twin.height = self.width, self.height
        twin.gap, twin.spacing = self.gap, self.spacing
        twin.seed, twin.rng = self.seed, random.Random(0)  # spawns beyond the horizon
        twin.y, twin.vy = self.y, self.vy
        twin.pipes = [dict(p) for p in self.pipes]
        twin.score, twin.steps, twin.frames = self.score, self.steps, self.frames
        twin.alive, twin.death_reason = True, None
        return twin

    def safe(self, flap: bool) -> bool:
        """Does this action survive the step, leaving a survivable next step (2-ply)?"""
        sim = self._clone()
        if not sim.step(flap):
            return False
        return any(sim._clone().step(second) for second in (True, False))

    def plan(self):
        """Facts for the prompt plus the deterministic autopilot recommendation."""
        pipe = self.next_pipe()
        offset = self.y - pipe["gap_y"]  # positive = bird below the gap center
        distance = pipe["x"] - BIRD_X
        safe_flap, safe_hold = self.safe(True), self.safe(False)
        if safe_flap != safe_hold:
            autopilot = "up" if safe_flap else "down"
        else:
            autopilot = "up" if offset > 0.35 else "down"
        return {
            "offset": offset,
            "distance": distance,
            "rising": self.vy < 0,
            "safe_flap": safe_flap,
            "safe_hold": safe_hold,
            "autopilot": autopilot,
        }

    def snapshot(self):
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "bird_x": BIRD_X,
            "bird_r": BIRD_R,
            "y": round(self.y, 3),
            "vy": round(self.vy, 3),
            "pipe_w": PIPE_W,
            "gap": self.gap,
            "pipes": [
                {"x": round(p["x"], 3), "gap_y": round(p["gap_y"], 3)}
                for p in self.pipes
                if p["x"] < self.width + 2
            ],
            "score": self.score,
            "steps": self.steps,
            "alive": self.alive,
            "death_reason": self.death_reason,
        }
