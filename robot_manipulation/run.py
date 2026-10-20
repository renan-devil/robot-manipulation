"""The sense -> think -> act loop, from the command line.

    python -m robot_manipulation.run                        # offline colour brain, seed 0
    python -m robot_manipulation.run --brain gemini --gif out/run.gif
    python -m robot_manipulation.run --instruction "Put the green cube on the blue tray."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .brains import BRAINS, COLORS, Plan, make_brain
from .sim import SimRobot


def run(brain, seed: int = 0, instruction: str | None = None,
        out_dir: Path | None = None, gif: bool = False) -> dict:
    robot = SimRobot(seed, record=gif)
    instruction = instruction or robot.task

    # 1. Sense: one picture from the overhead camera.
    image = robot.observe().image
    # 2. Think: the brain turns picture + words into two points.
    plan = brain.plan(image, instruction)
    # 3. Act: points -> 3D positions (depth ray) -> joint motions (inverse kinematics).
    grabbed = robot.pick_and_place(robot.normalized_to_world(plan.pick),
                                   robot.normalized_to_world(plan.place))

    result = {"seed": seed, "instruction": instruction, "pick": plan.pick, "place": plan.place,
              "grabbed": grabbed, "on_tray": robot.cubes_on_tray(), "brain_said": plan.raw}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _annotate(image, plan).save(out_dir / f"seed{seed}_brain_view.png")
        if gif and robot.frames:
            frames = [Image.fromarray(f) for f in robot.frames]
            frames[0].save(out_dir / f"seed{seed}.gif", save_all=True, append_images=frames[1:],
                           duration=60, loop=0)
    return result


def _annotate(image, plan: Plan) -> Image.Image:
    """The camera image with the brain's two points drawn on it: what did it 'see'?"""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for (y, x), color, label in ((plan.pick, "magenta", "pick"), (plan.place, "cyan", "place")):
        u, v = x / 1000 * img.width, y / 1000 * img.height
        draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=color, width=3)
        draw.text((u + 10, v - 6), label, fill=color)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", choices=sorted(BRAINS), default="color")
    parser.add_argument("--model", help="override the brain's default model id")
    parser.add_argument("--seed", type=int, default=0, help="which random scene")
    parser.add_argument("--episodes", type=int, default=1, help="run seeds seed..seed+N-1 and report a success rate")
    parser.add_argument("--instruction", help="default: the scene's own task")
    parser.add_argument("--out", type=Path, default=Path("out"), help="where to save images")
    parser.add_argument("--gif", action="store_true", help="also save a video of each episode")
    args = parser.parse_args()

    brain = make_brain(args.brain, args.model)
    successes = 0
    for seed in range(args.seed, args.seed + args.episodes):
        r = run(brain, seed, args.instruction, args.out, args.gif)
        # Success: exactly the cube named first in the instruction ended up on the tray.
        named = [w for w in r["instruction"].lower().replace(".", " ").split() if w in COLORS]
        ok = bool(named) and r["on_tray"] == [named[0] + "_cube"]
        successes += ok
        print(f"seed {seed}: {'SUCCESS' if ok else 'FAIL'}  {r['instruction']!r}  "
              f"pick={_fmt(r['pick'])} place={_fmt(r['place'])}  on tray: {r['on_tray']}")
    print(f"\n{successes}/{args.episodes} succeeded with the {args.brain} brain. Images in {args.out}/")


def _fmt(point) -> str:
    return "[{:.0f}, {:.0f}]".format(*point)


if __name__ == "__main__":
    main()
