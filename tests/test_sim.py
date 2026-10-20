import numpy as np
import pytest

from robot_manipulation.sim import CUBES, SimRobot, TOOL_DOWN


@pytest.fixture(scope="module")
def robot():
    return SimRobot(seed=0, width=320, height=240)


@pytest.mark.parametrize("target", [(0.25, 0.25, 0.03), (0.48, -0.25, 0.03), (0.3, 0.0, 0.2), (0.45, 0.2, 0.1)])
def test_ik_reaches_workspace_with_cup_down(robot, target):
    q, err = robot.solve_ik(np.array(target))
    assert err < 0.002
    d = robot.data
    saved = d.qpos.copy()
    d.qpos[robot.qadr] = q
    import mujoco
    mujoco.mj_kinematics(robot.model, d)
    axis = d.site_xmat[robot.tool].reshape(3, 3)[:, 2]
    d.qpos[:] = saved
    mujoco.mj_kinematics(robot.model, d)
    assert axis @ TOOL_DOWN > 0.99  # cup within ~8 degrees of vertical


def test_pixel_and_world_round_trip(robot):
    for name in CUBES:
        top = robot.cube_pos(name) + [0, 0, 0.025]
        u, v = robot.world_to_pixel(top)
        assert np.linalg.norm(robot.pixel_to_world(u, v) - top) < 0.003


def test_scenes_differ_by_seed():
    a, b = SimRobot(seed=1, width=64, height=48), SimRobot(seed=2, width=64, height=48)
    assert not np.allclose(a.cube_pos("red_cube"), b.cube_pos("red_cube"))


def test_pick_and_place_moves_the_cube(robot):
    robot.reset(0)
    start = robot.cube_pos("red_cube")
    tray = robot.model.body("blue_tray").pos.copy()
    assert robot.pick_and_place(start + [0, 0, 0.025], tray + [0, 0, 0.005])
    assert robot.cubes_on_tray() == ["red_cube"]


def test_missing_the_cube_grabs_nothing(robot):
    robot.reset(0)
    tray = robot.model.body("blue_tray").pos.copy()
    empty = tray + [0, 0, 0.005]  # nothing to pick on the tray itself
    assert not robot.pick_and_place(empty, empty)
    assert robot.cubes_on_tray() == []
