"""Hardware-free tests for the paddle duel planner."""

from laya_axera.paddle.planner import moved, plan, predict_x

COURT = {"width": 12, "length": 18, "ai_y": 1.2, "player_y": 16.8}
AI = {"x": 6, "w": 2.6, "step": 2}


def ball(x, vx=0.0, vy=-9.0, y=9.0):
    return {"x": x, "y": y, "vx": vx, "vy": vy, "r": 0.28}


def test_predicts_rail_bounces():
    x, _ = predict_x(COURT, ball(10, vx=9, vy=-9))
    assert 0.28 <= x <= 12 - 0.28
    straight, _ = predict_x(COURT, ball(4))
    assert abs(straight - 4) < 1e-6


def test_ball_moving_away_is_expected_back():
    x, eta = predict_x(COURT, ball(4, vy=9))
    assert abs(x - 4) < 1e-6 and eta > 1.0


def test_moves_toward_the_landing_point():
    assert plan(COURT, ball(9), AI)["best"] == "right"
    assert plan(COURT, ball(3), AI)["best"] == "left"


def test_a_move_stops_on_the_landing_point():
    assert moved(6, "right", 4, 6.9, 1.3, 10.7) == 6.9
    assert moved(6, "right", 4, 2.0, 1.3, 10.7) == 10.0


def test_holds_inside_the_dead_band_and_marks_the_rail():
    assert plan(COURT, ball(6.1), AI)["best"] == "hold"
    facts = plan(COURT, ball(11), {"x": 10.7, "w": 2.6, "step": 2})
    assert facts["tiers"]["right"] == "wall"
