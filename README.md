# LumengineEnvs

Reinforcement-learning **environments** for the Lumengine engine — the IsaacGymEnvs
analogue. The engine ships the `lm.rl` façade (the `RlSim` tensor API, `author_world`,
`create_world`, and the `rl_games` `train`/`play` trainer); this repo holds the
**tasks**, robot assets and training scripts that *use* it. Keeping them separate
mirrors Isaac Sim (engine) vs IsaacGymEnvs / Isaac Lab (envs): different deps
(torch / rl_games), asset licensing, and iteration cadence.

## Layout

```
tasks/
  _bootstrap.py        locate the engine's deployed lm.rl (LUMENGINE_ROOT)
  anymal_task.py       real anymal_c, velocity-command locomotion (IsaacGym parity)
  prep_anymal_usd.py   float the converted anymal + add PD drives (idempotent)
  cartpole_task.py     fixed-base cartpole reference task
assets/
  cartpole_converted/  committed (small, license-free)
  anymal_converted/    gitignored — regenerate (see below)
runs/                  gitignored — rl_games checkpoints
```

## Setup

1. Build the **Lumengine engine** so `lm.rl` is deployed to `build/<cfg>/python`.
2. Point this repo at it (or keep it a sibling `Lumengine2` next to `LumengineEnvs`):
   `set LUMENGINE_ROOT=...\Lumengine2`
3. GPU batch needs the shared CUDA context: `set LM_PHYSX_SHARE_CUDA_CONTEXT=1`.

## Anymal asset (regenerate — not vendored)

The real anymal_c is converted from the Isaac URDF:

```
PYTHONPATH= python -m urdf_usd_converter --no-physics-scene ^
  <IsaacGymEnvs>/assets/urdf/anymal_c/urdf/anymal.urdf assets/anymal_converted
```

`prep_anymal_usd.py` then makes it floating-base + PD-driven (the task runs it
automatically). Cartpole's asset is committed, so it works out of the box.

## Train / play

```
python tasks/anymal_task.py                                   # headless train -> runs/.../nn/*.pth
LM_RL_VIEW=1 python tasks/anymal_task.py                       # windowed train
LM_RL_PLAY=runs/<run>/nn/Anymal.pth python tasks/anymal_task.py   # replay (live vx/vy/yaw drive UI)
```

The anymal task is IsaacGymEnvs `Anymal` parity (48 obs, velocity-command exp-tracking
reward) and uses the `[256,128,64]` AnymalPPO network — the engine's default rl_games
config is cartpole-sized and won't learn a gait.
