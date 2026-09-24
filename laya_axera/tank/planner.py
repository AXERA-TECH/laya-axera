"""Grid facts for the AI tank: threats, lines of fire, and a route to a firing tile.

The arena is a grid of tiles (0 empty, 1 brick, 2 steel). Tanks sit on whole
tiles and face one of four compass directions; firing always goes the way the
tank faces, so "turn toward X and fire" is expressed as choosing direction X
with mode "shoot".
"""

import heapq

EMPTY, BRICK, STEEL = 0, 1, 2
ORDER = ("north", "south", "west", "east")
DIRS = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
THREAT_HORIZON = 6
BRICK_COST = 3
DANGER_COST = 6


def _inside(grid, x, y):
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def clear_line(grid, a, b):
    """a and b share a row or column and every tile strictly between is empty."""
    (ax, ay), (bx, by) = a, b
    if ax == bx and ay != by:
        lo, hi = sorted((ay, by))
        return all(grid[y][ax] == EMPTY for y in range(lo + 1, hi))
    if ay == by and ax != bx:
        lo, hi = sorted((ax, bx))
        return all(grid[ay][x] == EMPTY for x in range(lo + 1, hi))
    return False


def direction_to(a, b):
    (ax, ay), (bx, by) = a, b
    if ax == bx and ay != by:
        return "south" if by > ay else "north"
    if ay == by and ax != bx:
        return "east" if bx > ax else "west"
    return None


def _ray(grid, x, y, dx, dy, limit):
    """Empty tiles from (x, y) outward, stopping at the first wall or the edge."""
    tiles = []
    for _ in range(limit):
        if not _inside(grid, x, y) or grid[y][x] != EMPTY:
            break
        tiles.append((x, y))
        x, y = x + dx, y + dy
    return tiles


def threats(grid, bullets):
    """Tiles a player bullet will cross soon -> the direction the bullet comes from."""
    hit = {}
    for b in bullets:
        if b.get("owner") != "player":
            continue
        dx, dy = int(b["dx"]), int(b["dy"])
        x, y = int(round(b["x"])), int(round(b["y"]))
        for tile in _ray(grid, x, y, dx, dy, THREAT_HORIZON + 1):
            hit.setdefault(tile, (-dx, -dy))
    return hit


def firing_tiles(grid, target):
    """Tiles with a clear straight line to the target."""
    tx, ty = target
    tiles = set()
    for dx, dy in DIRS.values():
        tiles.update(_ray(grid, tx + dx, ty + dy, dx, dy, max(len(grid), len(grid[0]))))
    return tiles


def route(grid, start, goals, blocked, danger):
    """Cheapest first step toward any goal; bricks are passable at a cost (shoot, then go)."""
    if start in goals:
        return None
    best = {start: 0}
    first = {start: None}
    queue = [(0, start)]
    while queue:
        cost, (x, y) = heapq.heappop(queue)
        if (x, y) in goals:
            return first[(x, y)]
        if cost > best[(x, y)]:
            continue
        for name in ORDER:
            dx, dy = DIRS[name]
            nx, ny = x + dx, y + dy
            if not _inside(grid, nx, ny) or grid[ny][nx] == STEEL or (nx, ny) in blocked:
                continue
            step = 1 + (BRICK_COST if grid[ny][nx] == BRICK else 0)
            step += DANGER_COST if (nx, ny) in danger else 0
            if cost + step < best.get((nx, ny), float("inf")):
                best[(nx, ny)] = cost + step
                first[(nx, ny)] = first[(x, y)] or name
                heapq.heappush(queue, (cost + step, (nx, ny)))
    return None


def plan(grid, ai, player, bullets):
    """Best direction, what choosing each direction means, and a tier per direction.

    Priority: step off an incoming bullet's line, else shoot along a clear line to
    the player, else take the cheapest route to a firing tile (shooting through
    bricks on the way).
    """
    here = (int(ai["x"]), int(ai["y"]))
    target = (int(player["x"]), int(player["y"])) if player.get("alive", True) else None
    incoming = threats(grid, bullets)
    aim = set()
    if target is not None and player.get("dir") in DIRS:
        dx, dy = DIRS[player["dir"]]
        aim = set(_ray(grid, target[0] + dx, target[1] + dy, dx, dy, THREAT_HORIZON))
    danger = set(incoming) | aim
    occupied = {target} if target is not None else set()

    def adjacent(name):
        dx, dy = DIRS[name]
        return (here[0] + dx, here[1] + dy)

    def passable(name):
        x, y = adjacent(name)
        return _inside(grid, x, y) and grid[y][x] == EMPTY and (x, y) not in occupied

    modes = {name: "move" for name in ORDER}
    best, reason = None, "wander"

    if here in incoming:
        come_from = incoming[here]
        dodges = [n for n in ORDER if passable(n) and adjacent(n) not in incoming]
        off_line = [n for n in dodges if DIRS[n][0] * come_from[0] + DIRS[n][1] * come_from[1] == 0]
        if off_line or dodges:
            best, reason = (off_line or dodges)[0], "dodge"
        elif ai.get("can_fire", True):
            best = next(n for n, v in DIRS.items() if v == come_from)
            reason = "counter"
            modes[best] = "shoot"

    if target is not None:
        toward = direction_to(here, target)
        if toward and clear_line(grid, here, target):
            modes[toward] = "shoot"
            if best is None:
                best, reason = toward, "aim"

    if best is None and target is not None:
        first = route(grid, here, firing_tiles(grid, target) - occupied, occupied, danger)
        if first is not None:
            best, reason = first, "chase"
            x, y = adjacent(first)
            if grid[y][x] == BRICK:
                modes[first], reason = "shoot", "breach"

    if best is None:
        free = [n for n in ORDER if passable(n)]
        best = free[0] if free else ai.get("dir", "north")

    tiers = {}
    for name in ORDER:
        if name == best:
            tiers[name] = "best"
        elif not passable(name):
            tiers[name] = "wall"
        elif adjacent(name) in danger:
            tiers[name] = "risk"
        else:
            tiers[name] = "wrong"
    return {"best": best, "reason": reason, "modes": modes, "tiers": tiers}
