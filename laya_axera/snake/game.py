"""Deterministic Snake rules and a food-seeking planner that keeps its tail in reach.

The planner is the classic safe-snake strategy: take the shortest path to the
food, but only if a virtual snake that follows it could still reach its own
tail afterwards; otherwise wander toward the tail to buy space. A move is
"safe" when, after making it, the head can still reach the tail. When the
snake has gone too long without eating it gets hungry: it accepts any food path
whose first step is safe, and wanders among its roomy safe moves at random
(seeded by the game), which breaks the circling loops a purely deterministic
tail-chaser can fall into.
"""

import random
from collections import deque
from dataclasses import dataclass

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


@dataclass(frozen=True)
class MoveInfo:
    direction: str
    legal: bool
    safe: bool
    reason: str
    eats: bool


class SnakeGame:
    def __init__(self, width=24, height=16, seed=7, initial_length=6):
        if min(width, height) < 4:
            raise ValueError("Board dimensions must be >= 4")
        self.width, self.height, self.seed = width, height, seed
        self.capacity = width * height
        if not 2 <= initial_length <= width // 2:
            raise ValueError("Initial length must be >= 2 and fit in half a row")
        self.initial_length = initial_length
        # Flat-index neighbour table: the searches below run several times per move.
        self._nbrs = [
            [(x + dx) + (y + dy) * width for dx, dy in VECTORS.values()
             if 0 <= x + dx < width and 0 <= y + dy < height]
            for y in range(height) for x in range(width)
        ]
        self.rng = random.Random(seed)
        hx, hy = width // 2, height // 2
        self.body = deque((hx - i, hy) for i in range(initial_length))
        self.score = self.ticks = self.since_meal = 0
        self.alive, self.won = True, False
        self.death_reason = None
        self.food = self._spawn_food()

    @property
    def head(self):
        return self.body[0]

    def _inside(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _spawn_food(self):
        occupied = set(self.body)
        empty = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in occupied]
        return self.rng.choice(empty) if empty else None

    def target(self, direction):
        dx, dy = VECTORS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def legal_reason(self, direction):
        cell = self.target(direction)
        if not self._inside(cell):
            return "wall"
        if len(self.body) > 1 and cell == self.body[1]:
            return "reverse"
        occupied = set(self.body)
        if cell != self.food:
            occupied.discard(self.body[-1])  # The tail moves on a non-growing step.
        return "body" if cell in occupied else "legal"

    # ---- search helpers on an explicit body (head first) ----

    def _index(self, cell):
        return cell[0] + cell[1] * self.width

    def _blocked(self, cells):
        mask = bytearray(self.capacity)
        for x, y in cells:
            mask[x + y * self.width] = 1
        return mask

    def _path(self, start, goal, blocked):
        """Shortest path from start to goal avoiding blocked cells, or None."""
        mask = self._blocked(blocked)
        s, t = self._index(start), self._index(goal)
        prev = [-1] * self.capacity
        prev[s] = s
        queue = deque([s])
        nbrs = self._nbrs
        while queue:
            i = queue.popleft()
            if i == t:
                path = []
                while i != s:
                    path.append((i % self.width, i // self.width))
                    i = prev[i]
                path.append(start)
                return path[::-1]
            for j in nbrs[i]:
                if prev[j] < 0 and not mask[j]:
                    prev[j] = i
                    queue.append(j)
        return None

    def _tail_reachable(self, body):
        if len(body) <= 3:
            return True
        cells = list(body)
        mask = self._blocked(cells[1:-1])
        s, t = self._index(cells[0]), self._index(cells[-1])
        seen = bytearray(self.capacity)
        seen[s] = 1
        queue = deque([s])
        nbrs = self._nbrs
        while queue:
            i = queue.popleft()
            for j in nbrs[i]:
                if j == t:
                    return True
                if not seen[j] and not mask[j]:
                    seen[j] = 1
                    queue.append(j)
        return False

    def _after(self, body, cell, food):
        moved = deque(body)
        moved.appendleft(cell)
        if cell != food:
            moved.pop()
        return moved

    def _area(self, body):
        """Free cells reachable from the head, a tie-breaker for wandering."""
        cells = list(body)
        mask = self._blocked(cells[1:])
        s = self._index(cells[0])
        seen = bytearray(self.capacity)
        seen[s] = 1
        queue = deque([s])
        count = 1
        nbrs = self._nbrs
        while queue:
            i = queue.popleft()
            for j in nbrs[i]:
                if not seen[j] and not mask[j]:
                    seen[j] = 1
                    count += 1
                    queue.append(j)
        return count

    # ---- planner ----

    def moves(self):
        if not self.alive or self.won:
            return []
        out = []
        for direction in DIRECTIONS:
            reason = self.legal_reason(direction)
            legal = reason == "legal"
            cell = self.target(direction)
            safe = legal and self._tail_reachable(self._after(self.body, cell, self.food))
            if legal and not safe:
                reason = "would lose the way back to the tail"
            out.append(MoveInfo(direction, legal, safe, reason, cell == self.food))
        return out

    def _direction_to(self, cell):
        dx, dy = cell[0] - self.head[0], cell[1] - self.head[1]
        return next(d for d, v in VECTORS.items() if v == (dx, dy))

    @property
    def hungry(self):
        return self.since_meal > max(60, 2 * len(self.body))

    def _route(self, direction):
        """A food path that starts with `direction`, as (body after the first step, path)."""
        cell = self.target(direction)
        body = self._after(self.body, cell, self.food)
        if cell == self.food:
            return body, [cell]
        return body, self._path(body[0], self.food, set(list(body)[:-1]))

    def _keeps_tail(self, body, path):
        """Whether the tail stays in reach at every step of the path, including after
        eating. The per-step check matches the per-move safety rule; checking only the
        end state lets a route pass that the next move's safety check then refuses, and
        the snake turns back into its coil."""
        if not self._tail_reachable(body):
            return False
        for step in path[1:]:
            body = self._after(body, step, self.food)
            if not self._tail_reachable(body):
                return False
        return True

    def preferred(self, moves=None):
        """The planner's move: the shortest food route that keeps the tail in reach, else
        the tail. Every safe first step is tried, so a tie between equally short paths
        never decides whether the snake eats."""
        moves = moves if moves is not None else self.moves()
        safe = [m.direction for m in moves if m.safe]  # DIRECTIONS order keeps ties deterministic
        if not safe:
            legal = [m.direction for m in moves if m.legal]
            return legal[0] if legal else None
        if self.food is not None:
            routes = []
            for order, d in enumerate(safe):
                body, path = self._route(d)
                if path:
                    routes.append((len(path), order, d, body, path))
            routes.sort(key=lambda r: r[:2])  # shortest first; ties keep DIRECTIONS order
            for _, _, d, body, path in routes:
                if self._keeps_tail(body, path):
                    return d
            if self.hungry and routes:
                shortest = routes[0][0]
                return self.rng.choice([d for n, _, d, _, _ in routes if n == shortest])
        # No safe way to the food yet: keep the most room and a long way back to the
        # tail. A hungry snake picks among the roomy safe moves at random (seeded, so a
        # game still replays exactly); a fixed choice here can repeat the same loop forever.
        scored = []
        for direction in safe:
            body = self._after(self.body, self.target(direction), self.food)
            to_tail = self._path(body[0], body[-1], set(list(body)[1:-1]))
            room = self._area(body)
            scored.append((direction, room, len(to_tail) if to_tail else 0, room >= len(body)))
        if self.hungry:
            roomy = [d for d, _, _, ok in scored if ok] or [d for d, *_ in scored]
            return self.rng.choice(roomy)
        return max(scored, key=lambda t: (t[1], t[2]))[0]

    def food_reachability(self):
        """Current empty-cell connectivity; the occupied tail is not treated as empty."""
        blocked = set(self.body) - {self.head}
        visited = {self.head}
        queue = deque([self.head])
        while queue:
            x, y = queue.popleft()
            for dx, dy in VECTORS.values():
                cell = x + dx, y + dy
                if self._inside(cell) and cell not in blocked and cell not in visited:
                    visited.add(cell)
                    queue.append(cell)
        return self.food in visited, len(visited)

    def step(self, direction):
        if not self.alive or self.won:
            raise RuntimeError("Cannot step a finished game")
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        self.ticks += 1
        reason = self.legal_reason(direction)
        if reason != "legal":
            self.alive, self.death_reason = False, reason
            return False
        target = self.target(direction)
        self.body.appendleft(target)
        self.since_meal += 1
        if target == self.food:
            self.score += 1
            self.since_meal = 0
            if len(self.body) == self.capacity:
                self.won, self.food = True, None
            else:
                self.food = self._spawn_food()
            return True
        self.body.pop()
        return False

    def snapshot(self):
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "body": [list(cell) for cell in self.body],
            "food": list(self.food) if self.food else None,
            "score": self.score,
            "length": len(self.body),
            "ticks": self.ticks,
            "alive": self.alive,
            "won": self.won,
            "death_reason": self.death_reason,
        }
