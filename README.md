# LumengineEnvs

Reinforcement-learning **environments** for the Lumengine engine — the IsaacGymEnvs
analogue. The engine ships the `lm.rl` façade (the `RlSim` tensor API, the `World` +
morph authoring layer, and the `rl_games` / `rsl_rl` / `skrl` trainers); this repo holds
the **tasks**, robot assets and training scripts that *use* it. Keeping them separate
mirrors Isaac Sim (engine) vs IsaacGymEnvs / Isaac Lab (envs): different deps
(torch / trainers), asset licensing, and iteration cadence.

## Layout

```
tasks/
  _bootstrap.py        locate the engine's deployed lm.rl (LUMENGINE_ROOT)
  anymal_task.py       real anymal_c, velocity-command locomotion (IsaacGym parity)
  cartpole_task.py     fixed-base cartpole reference task
assets/
  anymal_c.rl.yaml     committed prep config (float base + PD drives) for anymal
  cartpole_converted/  committed (small, license-free)
  anymal_converted/    gitignored — regenerate (see below)
runs/                  gitignored — trainer checkpoints
tests/                 contract + smoke tests (skip cleanly without CUDA / a trainer)
```

## Setup

1. Build the **Lumengine engine** so `lm.rl` is deployed to `build/<cfg>/python`.
2. Point this repo at the engine (required — no implicit discovery):
   `set LUMENGINE_ROOT=C:\path\to\Lumengine2`
3. GPU batch needs the shared CUDA context: `set LM_PHYSX_SHARE_CUDA_CONTEXT=1`.
4. Trainers are optional and lazily imported; install what you use (see the CUDA-torch
   note in the file): `pip install -r requirements-rl.txt`.

## Anymal asset (regenerate — not vendored)

The real anymal_c is converted from the Isaac URDF (ANYbotics license — not committed):

```
PYTHONPATH= python -m urdf_usd_converter --no-physics-scene ^
  <IsaacGymEnvs>/assets/urdf/anymal_c/urdf/anymal.urdf assets/anymal_converted
```

No post-process script is needed: the task authors the robot through `rl.World` +
`rl.Usd(..., prep=True, config="anymal_c.rl.yaml")`, and `lm.rl`'s config-driven prep
floats the base + authors the PD `DriveAPI` from `assets/anymal_c.rl.yaml` (the prepped
USD is cached, the converted USD stays pristine). Cartpole's asset is committed, so it
works out of the box.

## Train / play

Pick the trainer with `LM_RL_TRAINER` (`rl_games` default, or `rsl_rl` / `skrl`):

```
python tasks/anymal_task.py                                   # headless train (rl_games)
LM_RL_TRAINER=rsl_rl python tasks/anymal_task.py              # rsl_rl (locomotion standard)
LM_RL_VIEW=1 python tasks/anymal_task.py                      # windowed train
LM_RL_INSTANCE=1 LM_RL_PLAY=runs/<ckpt> python tasks/anymal_task.py   # replay, instanced fleet
```

Checkpoint paths differ per trainer (`runs/.../nn/*.pth` for rl_games, `runs/model_*.pt`
for rsl_rl, `runs/.../checkpoints/` for skrl) — replay with the **same** `LM_RL_TRAINER`
it was trained with.

The anymal task is IsaacGymEnvs `Anymal` parity (48 obs, velocity-command exp-tracking
reward). On rl_games it uses the `[256,128,64]` AnymalPPO network (the engine's default
rl_games config is cartpole-sized and won't learn a gait); rsl_rl/skrl bring their own.
