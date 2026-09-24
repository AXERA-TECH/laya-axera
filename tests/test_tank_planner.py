"""Hardware-free tests for the tank duel planner."""

from laya_axera.tank.planner import BRICK, STEEL, clear_line, plan

N = 13


def arena():
    return [[0] * N for _ in range(N)]


def test_clear_shot_means_shoot_that_way():
    facts = plan(arena(), {"x": 6, "y": 2, "dir": "west"}, {"x": 6, "y": 10, "dir": "west"}, [])
    assert facts["best"] == "south" and facts["reason"] == "aim"
    assert facts["modes"]["south"] == "shoot"


def test_walls_block_the_line():
    grid = arena()
    grid[6][6] = STEEL
    assert not clear_line(grid, (6, 2), (6, 10))
    facts = plan(grid, {"x": 6, "y": 2, "dir": "south"}, {"x": 6, "y": 10, "dir": "west"}, [])
    assert facts["reason"] == "chase" and facts["modes"][facts["best"]] == "move"


def test_breach_a_lone_brick_when_it_is_cheapest():
    grid = arena()
    for x in range(N):
        grid[6][x] = STEEL
    grid[6][6] = BRICK
    facts = plan(grid, {"x": 6, "y": 5, "dir": "south"}, {"x": 6, "y": 10, "dir": "west"}, [])
    assert facts["best"] == "south" and facts["reason"] == "breach"
    assert facts["modes"]["south"] == "shoot"


def test_dodge_steps_off_the_bullet_line():
    bullet = {"x": 6, "y": 5.4, "dx": 0, "dy": -1, "owner": "player"}
    facts = plan(arena(), {"x": 6, "y": 2, "dir": "south"}, {"x": 2, "y": 10, "dir": "north"}, [bullet])
    assert facts["reason"] == "dodge" and facts["best"] in ("west", "east")


def test_own_bullets_are_not_threats():
    bullet = {"x": 6, "y": 5.4, "dx": 0, "dy": -1, "owner": "ai"}
    facts = plan(arena(), {"x": 6, "y": 2, "dir": "south"}, {"x": 2, "y": 10, "dir": "north"}, [bullet])
    assert facts["reason"] != "dodge"


def test_every_direction_gets_exactly_one_tier():
    grid = arena()
    grid[1][6] = STEEL
    facts = plan(grid, {"x": 6, "y": 2, "dir": "south"}, {"x": 9, "y": 9, "dir": "north"}, [])
    assert sorted(facts["tiers"]) == ["east", "north", "south", "west"]
    assert list(facts["tiers"].values()).count("best") == 1
    assert facts["tiers"]["north"] in ("wall", "best")
