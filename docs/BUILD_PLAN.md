# Build plan

**Goal:** control a robot arm remotely, by instruction, with AI doing the perception and decisions. Each phase spends a little more money than the previous one, and only once the previous phase has proven its point.

**Principle:** the brain/body split (see README). Each phase swaps one side and keeps the other, so a failure always has exactly one suspect.

| Phase | What | Spend | Status |
|---|---|---|---|
| 0 | Simulated arm + offline brain + tests | €0 | **Done** |
| 1 | Cloud brains (Gemini Robotics-ER, Mistral) on harder instructions | API usage, likely a few euros | Code done; needs API keys |
| 2 | Same brains, remote real robot | €0 to try, plus engineering time | Next |
| 3 | Own desk robot (SO-101 arm + camera) | A few hundred euros of hardware (*to verify*) | Decision gate |
| 4 | Learned policies (VLA: openpi π0.5 or NVIDIA GR00T) | GPU rental, roughly a few hundred euros per fine-tuning campaign (*to verify*) | Decision gate |

## Phase 0: foundations (done)

Built and pushed in this repository:

- A MuJoCo scene: a 4-joint arm with a suction cup, three coloured cubes, a blue tray, an overhead camera. Each random seed shuffles the scene and picks a task such as "Put the yellow cube on the blue tray."
- The body (`sim.py`):
  - **Pixel → 3D**: a ray from the camera's optical centre through the pixel; the first surface it hits is the target. This is pinhole-camera geometry: the pixel offset from the image centre divided by the focal length gives the ray's direction.
  - **3D → joints**: inverse kinematics by damped least squares, `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`, solving for position and keeping the cup vertical.
  - **Motion**: smoothstep interpolation between joint configurations, and pick/place primitives (approach from above, descend, grip, lift).
- The brain interface (`brains.py`), one shared answer format (`[y, x]` in 0-1000), three brains.
- 19 tests (about 3 seconds, no network): kinematics accuracy, camera round trip, end-to-end success, the exact HTTP requests the cloud brains send.

**Result:** the offline colour brain succeeds 10 out of 10 on random scenes. The pipeline works; any failure from now on comes from the brain or the real world, not the plumbing.

## Phase 1: cloud brains (next action: get two API keys)

1. Create a Gemini API key (Google AI Studio) and a Mistral key. Put them in your environment, never in the repository.
2. Run 20 episodes per brain: `python -m robot_manipulation.run --brain gemini --episodes 20`.
3. Add instructions the colour brain cannot follow: spatial ("the cube nearest the robot"), relational ("the cube left of the yellow one"), negation ("any cube that is not red").
4. Compare success rate, pointing error in millimetres, latency and cost per call.

**Decision gate:** if Gemini ER succeeds on at least ~80% of the harder instructions, the "point, then move" approach is good enough for phase 2. If not, read the failure images in `out/` before going further.

## Phase 2: a real robot over the internet

Two options, in order of preference:

- **AutoEval** (free, WidowX arms, queue with live stream). It expects a closed-loop policy that outputs small arm movements step by step, not two target points. Two ways to bridge that:
  - (a) Submit an open VLA, the π0.5-DROID checkpoint from openpi. Low effort, but it needs a rented GPU while the job runs.
  - (b) Write a small adapter: ask Gemini ER for the target point in the first image, then servo towards it. The open problem is converting pixels to metres without their camera calibration. More engineering, less GPU.
  Confirm first, on their Discord, which 2026 tasks are open.
- **CloudGripper** (KTH): API access to small robot cells; ask the lab or join their competition.

**Deliverable:** one recorded real-robot run driven by our code, with its success rate over 10 tries.

## Phase 3: own desk robot (decision gate: buy hardware?)

- Hardware: a LeRobot **SO-101** follower arm plus a leader arm (for teaching by demonstration) and a USB camera. The SO-101 model in MuJoCo Menagerie replaces our toy arm, so code is first validated in simulation against the exact same geometry.
- Software: a `LeRobotSO101` body with the same methods as `SimRobot`. The brains do not change.
- Camera calibration (a printed checkerboard) gives the pixel → 3D mapping that is perfect in simulation.
- "Remote" then means anyone on the team can send it instructions through a small web API, with a queue so only one job runs at a time. That queue is a simple first-in-first-out list in front of the robot.

**Why buy:** remote academic robots give you a few fixed tasks; your own arm gives you your own tasks (for example, a portfolio company's real use case) and unlimited data collection.

## Phase 4: learned skills (decision gate: rent GPUs?)

When a task is too delicate for "point, then move" (folding, inserting, handling soft objects):

1. Record 50 to 200 demonstrations with the leader arm (LeRobot does this).
2. Fine-tune an open VLA: openpi π0.5 (Physical Intelligence) or NVIDIA GR00T N1.6. Rent an A100 or H100 by the hour.
3. Evaluate in simulation first, then on the SO-101, then submit to AutoEval or RoboArena for an independent score.

## What this plan deliberately does not do yet

- Build a custom web dashboard: a command line is enough until several people use the robot.
- Use ROS 2: it is the industry standard for larger robots, but plain Python is simpler for one arm. Reconsider at phase 3 if we connect to industrial equipment.
- Use NVIDIA Isaac Sim before phase 4: it needs an RTX GPU and adds weight we do not need for one arm and three cubes.
