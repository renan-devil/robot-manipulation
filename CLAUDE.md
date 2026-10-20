# Working in this repository

## Co-authorship (Renan's convention)

Every commit pushed here is co-authored by Renan and Claude.

- Author: `Renan <renan@oss.ventures>`. Set it per clone with `git config user.name "Renan"` and `git config user.email renan@oss.ventures`.
- Every commit message ends with a `Co-Authored-By: Claude ...` trailer.
- `.githooks/commit-msg` adds that trailer when it is missing. Enable it once per clone with `git config core.hooksPath .githooks`.

## Architecture rule

Keep the brain/body split (see README): brains take an image and an instruction and return points in `[y, x]` 0-1000 image coordinates; bodies turn points into motion. A new model is a new class in `brains.py`; a new robot (real or remote) is a new body with the same methods as `SimRobot` (`observe`, `normalized_to_world`, `pick_and_place`, `cubes_on_tray` or its equivalent check).

## Before pushing

Run `pytest`. It must pass without network access or API keys: cloud brains are tested with a fake HTTP call.
