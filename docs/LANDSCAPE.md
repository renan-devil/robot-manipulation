# Landscape: AI for robot arms, October 2026

What exists, who it is from, what it costs to try, and whether this project uses it. Facts were checked on the web on 20 October 2026. Where sources were thin or disagreed, the entry says so. Check anything marked *to verify* before spending money on it.

## Two ways to put intelligence in a robot

1. **A model that points, plus classic robotics that moves.** A vision-language model looks at the image and returns *where* (a pixel, a box, a trajectory). Inverse kinematics and motion planning turn that into joint motion. It needs no robot training data and works with today's APIs. Its limit is that it only handles tasks you can describe as "move this there".
2. **A vision-language-action (VLA) model.** One neural network takes images, joint angles and an instruction, and outputs motor commands many times a second. It can learn dexterous skills such as folding or inserting, but it needs demonstration data from your robot and a GPU to fine-tune and run.

This repository starts with approach 1 (phases 0 to 2 of the [build plan](BUILD_PLAN.md)) and adds approach 2 later (phase 4).

## Google DeepMind

| What | Status | Use here |
|---|---|---|
| **Gemini Robotics-ER 2** (`gemini-robotics-er-2-preview`, plus a `-streaming-preview` variant for the Live API) | Preview on the Gemini API since 30 July 2026. Input: text, image, video, audio. Points come back as `[y, x]` normalized to 0-1000. ER 1.6 was scheduled to shut down on 31 August 2026. Pricing is not on the model page; *to verify* on the Gemini pricing page. One third-party guide calls it a closed preview, while the official page links straight to AI Studio. *To verify* with a key. | **Yes**: `GeminiBrain`, the default cloud brain |
| **Gemini Robotics** (the VLA that drives motors) and its on-device version | Access through a trusted-tester programme, not a public API | Later, if access is granted |
| **MuJoCo** | Open-source physics simulator (Apache 2.0), version 3.16 on PyPI | **Yes**: the simulator under `sim.py` |
| **MuJoCo Menagerie** | Open library of accurate robot models (Franka, UR5e, SO-101, ALOHA and others) | Phase 3, to swap the toy arm for the real arm's model |

## Mistral AI

| What | Status | Use here |
|---|---|---|
| **Robostral Navigate** | Mistral's first robotics model, announced on 7 or 8 July 2026 (sources differ on the day). Reported as 8B parameters, working from one RGB camera and plain-language instructions, trained in simulation. It is for **navigation** (moving a mobile robot), not for arm manipulation. Benchmark claims are Mistral's own. | No: it is not a manipulation model |
| **Mistral vision models** (Medium, Small, Pixtral) | Public API with image input | **Yes**: `MistralBrain`. General models, not trained to point precisely, so we expect lower accuracy than Gemini ER. Measuring that gap is a phase 1 deliverable. |

## NVIDIA

| What | Status | Use here |
|---|---|---|
| **Isaac GR00T N1.6** (open VLA, 3B parameters, `nvidia/GR00T-N1.6-3B` on Hugging Face) | Open weights. N1.7 reported in 2026; one source calls it generally available, the GitHub page I found says early access. *To verify.* Built for humanoids, fine-tunable for arms. | Phase 4 candidate |
| **Isaac Sim / Isaac Lab** | Open-source GPU simulator and robot-learning framework. Needs an NVIDIA RTX GPU. | Phase 4, to train or fine-tune at scale |
| **Cosmos** | World models used to generate training data | Not planned |

## Others worth knowing

| What | Status | Use here |
|---|---|---|
| **Physical Intelligence openpi** (π0, π0-FAST, π0.5) | Open weights, Apache 2.0, including a π0.5 checkpoint fine-tuned on the DROID dataset. Newer models (π0.6, π0.7) stay closed. Inference needs a GPU with more than 8 GB; full fine-tuning needs more than 70 GB (A100 or H100 class). | Phase 4 candidate; the most likely policy to submit to AutoEval |
| **Hugging Face LeRobot** and the **SO-101** arm | Open-source library plus a low-cost 6-joint arm you 3D-print or buy as a kit; integrates openpi and Hugging Face's small SmolVLA model. Kit prices *to verify*; budget a few hundred euros for a leader + follower pair. | Phase 3: the recommended desk robot |

## Real robots you can use remotely

This is what you heard about as "a queue to use a robot arm online". There are a few, all academic, none of them a commercial pay-per-minute API.

| What | How it works | Fit |
|---|---|---|
| **AutoEval** (UC Berkeley / Stanford) | Two stations with WidowX arms that reset their own scenes and detect success automatically. You run a *policy server* on your own machine, expose it with ngrok or bore, and submit its address on their dashboard. Jobs wait in a queue, with a live stream. Protocol: they send a 256×256 image, an 8-number robot state and the instruction; you answer with a 7-number end-effector movement, step by step. The four public tasks (open or close a drawer, put an eggplant in a basket or a sink) were announced as available until 1 January 2026, with 2026 tasks "TBD". **Current availability to verify** on their Discord. | Best match. It needs a closed-loop policy (approach 2), not just a pointing model. |
| **CloudGripper** (KTH, Sweden) | Open-source rack of small robot cells accessible over an API; used for the ICRA 2026 Cloud Manipulation Competition | Good for testing; access via the competition or the lab |
| **RoboArena** | Distributed evaluation on DROID (Franka) robots: volunteer labs run pairwise, blind comparisons of submitted policies | For benchmarking a VLA, not for interactive use |
| **The Construct** | Book a real UR3 arm in their remote lab through a web interface, without an API; Enterprise plan only | Teaching, not automation |

## Sources

- [Gemini Robotics ER 2 Preview, model page](https://ai.google.dev/gemini-api/docs/models/gemini-robotics-er-2-preview)
- [Gemini API, spatial reasoning for robotics](https://ai.google.dev/gemini-api/docs/robotics-spatial)
- [Gemini Robotics ER 2 explained (third party)](https://www.orcarouter.ai/blog/gemini-robotics-er-2-explained)
- [Mistral launches first robotics model (Reuters, via The Star)](https://www.thestar.com.my/tech/tech-news/2026/07/08/mistral-launches-first-robotics-model-in-physical-ai-push)
- [Mistral Robostral Navigate (Crypto Briefing)](https://cryptobriefing.com/mistral-robostral-navigate-robotics-model/)
- [ThursdAI, Mistral AI](https://thursdai.news/companies/mistral-ai)
- [NVIDIA Isaac GR00T N1.7 and Newton explained](https://codeforgeek.com/nvidias-new-ai-robots-gr00t-n1-newton-physics-engine/)
- [Isaac GR00T N1.7 GitHub page (third-party fork)](https://github.com/precognitionlab/Isaac-GR00T-N1.7)
- [NVIDIA humanoid reference design, GTC Taipei 2026](https://humanoid.guide/?p=11909)
- [openpi overview](https://aiwiki.ai/wiki/openpi) and [Physical Intelligence](https://www.pi.website/)
- [AutoEval project page](https://auto-eval.github.io/) and [paper](https://arxiv.org/abs/2503.24278v2)
- [RoboArena](https://awesomepapers.io/papers/atreya2025roboarena)
- [CloudGripper / Open Source Cloud Robotic Manipulation](https://www.epfl.ch/labs/lasa/?p=5971)
- [The Construct remote lab](https://www.theconstruct.ai/?p=29677)
