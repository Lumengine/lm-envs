# LumengineEnvs

Reinforcement-learning **environments** for the [Lumengine](https://lumengine.com)
engine — the IsaacGymEnvs / Isaac Lab / Genesis analogue. The engine ships the
`lm.rl` façade (the `RlSim` GPU-tensor API, the `World` + morph USD-authoring layer,
and the `rl_games` / `rsl_rl` / `skrl` trainers); this repo holds the **tasks**, robot
assets and a small task registry + CLI that *use* it.

Robots are authored from their **upstream URDF/MJCF** and converted to USD by the
engine — GPU-vectorized, thousands of environments in one PhysX scene.

## Quickstart

```bash
# 1. Build the Lumengine engine so lm.rl is deployed to build/<cfg>/python
# 2. Point this repo at the engine (required):
set LUMENGINE_ROOT=C:\path\to\Lumengine
# 3. GPU batch needs the shared CUDA context:
set LM_PHYSX_SHARE_CUDA_CONTEXT=1
# 4. Trainers are optional + lazily imported; install what you use:
pip install -r requirements-rl.txt

python train.py --list                  # the task catalog
python train.py --task Anymal           # headless train (rl_games)
python train.py --task Ant --view       # windowed: watch it train
python play.py  --task Anymal --checkpoint runs/Anymal_.../nn/Anymal.pth --cmd 1,0,0
```

## Tasks

| Task | Domain | Robot | Description |
|---|---|---|---|
| `Cartpole` | classic | cartpole | Balance a pole on a force-controlled cart. |
| `Ant` | locomotion | MuJoCo ant (MJCF) | Run forward as fast as possible (torque control). |
| `Anymal` | locomotion | ANYmal-C (URDF) | Velocity-command walking (IsaacLab-style reward). |

More robots and tasks (Unitree quadrupeds, humanoids, manipulators, dexterous hands,
drones) are landing — see `docs/plans/` / the roadmap.

## Train / play

The CLI is the front door; every task is registered in
[`lumengine_envs/registry.py`](lumengine_envs/registry.py).

```bash
python train.py --task <Name> [--num-envs N] [--epochs E] \
                [--trainer rl_games|rsl_rl|skrl] [--seed S] [--view]
python play.py  --task <Name> --checkpoint <path> [--num-envs 16] [--cmd vx,vy,yaw]
```

- `--view` trains/plays in a window; otherwise headless.
- Pick the trainer with `--trainer` (default `rl_games`). Replay a checkpoint with the
  **same** trainer it was trained with (`runs/.../nn/*.pth` for rl_games).
- Locomotion tasks accept a fixed command via `--cmd "vx,vy,yaw"` (e.g. `1,0,0` =
  everyone walks forward) — handy for a clean demo.

## Layout

```
train.py  play.py            unified CLI (registry-driven)
lumengine_envs/              the package: task registry (+ tasks as they migrate here)
tasks/                       task implementations (cartpole/ant/anymal) + _bootstrap
assets/                      robot sources (URDF/MJCF) + per-robot LICENSE; USD is a build artifact
configs/                     per-task config (as tasks migrate)
baselines/                   reproducible baselines (command + reward + checkpoint + gif)
tests/                       contract + smoke tests (skip cleanly without CUDA / a trainer)
runs/                        trainer checkpoints (gitignored)
```

## Assets & licensing

LumengineEnvs is **Apache-2.0** (see [`LICENSE`](LICENSE)). Robot models keep their
**own** licenses — every redistributed asset is listed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) with its license and origin, and
keeps its original license file under `assets/<robot>/`.

Models are sourced from their upstream URDF/MJCF descriptions and converted to USD
locally. **Nothing is taken from NVIDIA Isaac Sim / Omniverse Nucleus** (proprietary).
Permissively-licensed robots (BSD/MIT/Apache) are vendored; the rest are fetched on
demand by `scripts/fetch_assets.py`. The real ANYmal-C is fetched (not vendored) — its
source converts via `rl.Usd(..., prep=True, config="assets/anymal_c.rl.yaml")`.
