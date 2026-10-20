import json

import pytest

from robot_manipulation import brains
from robot_manipulation.brains import ColorBrain, GeminiBrain, MistralBrain, parse_plan
from robot_manipulation.run import run
from robot_manipulation.sim import SimRobot


def test_parse_plan_accepts_fences_and_point_objects():
    plan = parse_plan('```json\n{"pick": {"point": [400, 500], "label": "red cube"}, "place": [600, 300]}\n```')
    assert plan.pick == (400, 500) and plan.place == (600, 300)


def test_parse_plan_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_plan('{"pick": [1200, 5], "place": [1, 1]}')


def test_color_brain_points_at_the_right_cube():
    robot = SimRobot(seed=4, width=320, height=240)
    plan = ColorBrain().plan(robot.observe().image, "Put the red cube on the blue tray.")
    picked = robot.normalized_to_world(plan.pick)
    assert abs(picked[0] - robot.cube_pos("red_cube")[0]) < 0.01
    assert abs(picked[1] - robot.cube_pos("red_cube")[1]) < 0.01


@pytest.mark.parametrize("seed", range(5))
def test_end_to_end_with_offline_brain(seed):
    result = run(ColorBrain(), seed)
    colour = result["instruction"].split()[2]
    assert result["on_tray"] == [colour + "_cube"]


def _fake_post(reply, seen):
    def post(url, headers, body, timeout=120):
        seen.update(url=url, headers=headers, body=body)
        return reply
    return post


def test_gemini_request_and_reply(monkeypatch):
    seen = {}
    reply = {"candidates": [{"content": {"parts": [{"text": '{"pick": [100, 200], "place": [300, 400]}'}]}}]}
    monkeypatch.setattr(brains, "_post_json", _fake_post(reply, seen))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    plan = GeminiBrain().plan(SimRobot(width=64, height=48).observe().image, "Put the red cube on the blue tray.")
    assert plan.pick == (100, 200)
    assert "gemini-robotics-er" in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert seen["body"]["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png"


def test_mistral_request_and_reply(monkeypatch):
    seen = {}
    reply = {"choices": [{"message": {"content": json.dumps({"pick": [10, 20], "place": [30, 40]})}}]}
    monkeypatch.setattr(brains, "_post_json", _fake_post(reply, seen))
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    plan = MistralBrain().plan(SimRobot(width=64, height=48).observe().image, "Put the red cube on the blue tray.")
    assert plan.place == (30, 40)
    assert seen["headers"]["Authorization"] == "Bearer test-key"


def test_missing_key_explains_what_to_do(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiBrain().plan(SimRobot(width=64, height=48).observe().image, "x")
