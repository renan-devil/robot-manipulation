"""The robot's body: a simulated desk arm in MuJoCo.

Everything a real arm driver would offer lives here, behind a few methods:

    observe()          -> camera image + joint angles
    pixel_to_world()   -> where in the room a pixel is (like a depth camera)
    pick(), place()    -> motion primitives built on inverse kinematics

The "brain" (see brains.py) never touches MuJoCo. It only sees images and
returns points, so the same brain can later drive a real arm.

Two simplifications keep this first version small, both stated openly:
  * The arm is kinematic: we set its joint angles directly, as if it had a
    perfect position controller. Only the cubes are simulated physically.
  * The gripper is a suction cup: when it touches a cube, the cube follows
    the tool rigidly until released.
"""

from __future__ import annotations

import os
import sys

# Headless Linux (servers, CI, this cloud box) has no screen: render on the CPU.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSET = Path(__file__).parent / "assets" / "desk_arm.xml"
ARM_JOINTS = ("yaw", "shoulder", "elbow", "wrist")
CUBES = ("red_cube", "green_cube", "yellow_cube")
TRAY = "blue_tray"

TOOL_DOWN = np.array([0.0, 0.0, -1.0])  # the suction cup should face the table
HOME = np.array([0.05, 0.0, 0.45])       # parked above the base, out of the camera's way
CUBE_HALF = 0.025
TRAY_HALF = 0.07


@dataclass
class Observation:
    image: np.ndarray   # H x W x 3, uint8, from the overhead camera
    joints: np.ndarray  # arm joint angles in radians
    tool: np.ndarray    # tool tip position in metres


class SimRobot:
    def __init__(self, seed: int = 0, width: int = 640, height: int = 480, record: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.video_renderer = mujoco.Renderer(self.model, 240, 320) if record else None
        self.frames: list[np.ndarray] = []

        joints = [self.model.joint(n) for n in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.qlow = np.array([j.range[0] for j in joints])
        self.qhigh = np.array([j.range[1] for j in joints])
        self.tool = self.model.site("tool").id
        self.held: tuple[str, np.ndarray, np.ndarray] | None = None  # (cube, offset, rotation)
        self.task = self.reset(seed)

    # ------------------------------------------------------------------ scene

    def reset(self, seed: int = 0) -> str:
        """Shuffle the cubes and the tray; return a task instruction for this seed."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.held = None

        # Rejection sampling: random spots in the workspace, far enough apart.
        spots: list[np.ndarray] = []
        while len(spots) < 1 + len(CUBES):
            p = np.array([rng.uniform(0.25, 0.48), rng.uniform(-0.25, 0.25)])
            # Spot 0 is the tray, which is bigger than a cube: keep more room around it.
            if all(np.linalg.norm(p - s) > (0.17 if i == 0 else 0.11) for i, s in enumerate(spots)):
                spots.append(p)

        self.model.body(TRAY).pos[:] = [*spots[0], 0.005]
        for name, (x, y) in zip(CUBES, spots[1:]):
            yaw = rng.uniform(-np.pi, np.pi)
            adr = self.model.joint(self.model.body(name).jntadr[0]).qposadr[0]
            self.data.qpos[adr:adr + 7] = [x, y, CUBE_HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        self.q = self.solve_ik(HOME, q0=np.array([0.0, 0.3, 1.5, 1.3]))[0]
        self._set_arm(self.q)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):  # let the cubes settle on the table
            self._step(self.q)

        color = CUBES[rng.integers(len(CUBES))].split("_")[0]
        return f"Put the {color} cube on the blue tray."

    def observe(self) -> Observation:
        self.renderer.update_scene(self.data, camera="top")
        return Observation(self.renderer.render().copy(), self.q.copy(), self.tool_pos())

    def tool_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tool].copy()

    def cube_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def cubes_on_tray(self) -> list[str]:
        tray = self.model.body(TRAY).pos
        on = []
        for name in CUBES:
            p = self.cube_pos(name)
            inside = np.all(np.abs(p[:2] - tray[:2]) < TRAY_HALF)
            if inside and 0.01 < p[2] < 0.06:
                on.append(name)
        return on

    # ----------------------------------------------------------------- camera

    def _camera(self, name: str = "top"):
        cam = self.model.camera(name).id
        pos = self.data.cam_xpos[cam]
        rot = self.data.cam_xmat[cam].reshape(3, 3)  # columns = camera axes in world frame
        focal = 0.5 * self.height / np.tan(np.radians(self.model.cam_fovy[cam]) / 2)
        return pos, rot, focal

    def world_to_pixel(self, p: np.ndarray) -> tuple[float, float]:
        """Pinhole projection. The camera looks along its own -z axis."""
        pos, rot, f = self._camera()
        c = rot.T @ (np.asarray(p) - pos)
        u = self.width / 2 + f * c[0] / -c[2]
        v = self.height / 2 - f * c[1] / -c[2]
        return u, v

    def pixel_to_world(self, u: float, v: float) -> np.ndarray:
        """Shoot a ray through pixel (u, v) and return the first surface it hits.

        This plays the role of a depth camera. The robot itself (geom group 1)
        is transparent to the ray, so it can never pick a point on its own arm.
        """
        pos, rot, f = self._camera()
        ray = rot @ np.array([(u - self.width / 2) / f, -(v - self.height / 2) / f, -1.0])
        ray /= np.linalg.norm(ray)
        geom = np.zeros(1, dtype=np.int32)
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        dist = mujoco.mj_ray(self.model, self.data, pos, ray, groups, 1, -1, geom)
        if dist < 0:
            raise ValueError(f"pixel ({u:.0f}, {v:.0f}) does not look at anything")
        return pos + dist * ray

    def normalized_to_world(self, point_yx: tuple[float, float]) -> np.ndarray:
        """Convert a [y, x] point in 0-1000 image coordinates (Gemini's convention)."""
        y, x = point_yx
        return self.pixel_to_world(x / 1000 * self.width, y / 1000 * self.height)

    # --------------------------------------------------------------- kinematics

    def solve_ik(self, target: np.ndarray, q0: np.ndarray | None = None,
                 iters: int = 300, damping: float = 0.05) -> tuple[np.ndarray, float]:
        """Joint angles that put the tool tip at `target`, cup facing down.

        Damped least squares: with J the Jacobian (how the tool moves per unit
        of joint motion) and e the error, each iteration takes
            dq = J^T (J J^T + damping^2 I)^-1 e
        which is the least-squares step, kept small near singular poses.
        The error stacks position (3 rows) and orientation (3 rows: the
        rotation that would turn the tool axis onto straight down).
        Returns the angles and the remaining position error in metres.
        """
        m = self.model
        d = mujoco.MjData(m)  # scratch copy: solving must not move the real arm
        d.qpos[:] = self.data.qpos
        q = np.array(self.q if q0 is None else q0, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        w = 0.3  # how much we care about orientation versus position
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos_err = target - d.site_xpos[self.tool]
            axis = d.site_xmat[self.tool].reshape(3, 3)[:, 2]
            rot_err = np.cross(axis, TOOL_DOWN)
            if np.linalg.norm(pos_err) < 5e-4 and np.linalg.norm(rot_err) < 1e-2:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tool)
            J = np.vstack([jacp[:, self.dadr], w * jacr[:, self.dadr]])
            e = np.concatenate([pos_err, w * rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + dq, self.qlow, self.qhigh)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        return q, float(np.linalg.norm(target - d.site_xpos[self.tool]))

    # ----------------------------------------------------------------- motion

    def move_to(self, target, steps: int = 300) -> None:
        """Glide the tool tip to `target` along a smooth joint-space path."""
        goal, err = self.solve_ik(np.asarray(target, dtype=float))
        if err > 0.005:
            raise ValueError(f"target {np.round(target, 3)} is out of reach ({err * 1000:.0f} mm off)")
        start = self.q.copy()
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)  # smoothstep: start and stop gently
            self._step(start + s * (goal - start))
            if self.video_renderer is not None and i % 15 == 0:
                self.video_renderer.update_scene(self.data, camera="front")
                self.frames.append(self.video_renderer.render().copy())

    def pick(self, surface_point: np.ndarray) -> bool:
        """Approach from above, touch the surface, switch suction on, lift."""
        p = np.asarray(surface_point, dtype=float)
        self.move_to(p + [0, 0, 0.12])
        self.move_to(p + [0, 0, 0.003], steps=150)
        self._suction_on()
        self.move_to(p + [0, 0, 0.15], steps=200)
        return self.held is not None

    def place(self, surface_point: np.ndarray) -> None:
        """Carry the held cube above `surface_point`, lower it, release, back off."""
        p = np.asarray(surface_point, dtype=float)
        self.move_to(p + [0, 0, 0.15])
        if self.held is not None:
            self.move_to(p + [0, 0, 2 * CUBE_HALF + 0.008], steps=150)
        self.held = None
        self.move_to(p + [0, 0, 0.15], steps=150)

    def pick_and_place(self, pick_point: np.ndarray, place_point: np.ndarray) -> bool:
        grabbed = self.pick(pick_point)
        if grabbed:
            self.place(place_point)
        self.move_to(HOME)
        for _ in range(200):  # let the cube settle before anyone checks
            self._step(self.q)
        return grabbed

    # --------------------------------------------------------------- internals

    def _set_arm(self, q: np.ndarray) -> None:
        self.q = q
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.dadr] = 0

    def _step(self, q: np.ndarray) -> None:
        self._set_arm(q)
        mujoco.mj_step(self.model, self.data)
        self._set_arm(q)  # undo whatever gravity did to the arm during the step
        mujoco.mj_kinematics(self.model, self.data)
        if self.held is not None:
            self._carry()
        mujoco.mj_forward(self.model, self.data)

    def _suction_on(self) -> None:
        """Grab the cube whose top face the cup is touching, if any."""
        tip = self.tool_pos()
        for name in CUBES:
            body = self.data.body(name)
            local = body.xmat.reshape(3, 3).T @ (tip - body.xpos)  # tip in cube frame
            if np.all(np.abs(local) < CUBE_HALF + 0.012):
                rot_tool = self.data.site_xmat[self.tool].reshape(3, 3)
                offset = rot_tool.T @ (body.xpos - tip)
                rotation = rot_tool.T @ body.xmat.reshape(3, 3)
                self.held = (name, offset, rotation)
                return

    def _carry(self) -> None:
        """Pin the held cube to the tool, as a suction cup would."""
        name, offset, rotation = self.held
        rot_tool = self.data.site_xmat[self.tool].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rot_tool @ rotation).flatten())
        joint = self.model.joint(self.model.body(name).jntadr[0])
        self.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = [*(self.tool_pos() + rot_tool @ offset), *quat]
        self.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0
