# lm-envs

Reinforcement-learning **environments** for the [Lumengine](https://lumengine.com)
engine. The engine ships the `lm.rl` façade (the `RlSim` GPU-tensor API, the
`World` + morph USD-authoring layer, and the `rl_games` / `rsl_rl` / `skrl`
trainer adapters); this repo holds the **tasks**, robot assets and a small task
registry + CLI that *use* it.

Robots are authored from their **upstream URDF/MJCF** and converted to USD by the
engine — GPU-vectorized, thousands of environments in one PhysX scene.

## Quickstart

```bash
# 1. Build the Lumengine engine so lm.rl is deployed to build/<cfg>/python
# 2. Point this repo at the engine (required):
set LUMENGINE_ROOT=C:\path\to\Lumengine
# 3. Trainers are optional + lazily imported; install what you use:
pip install -r requirements-rl.txt
# 4. One asset is fetched, not committed (ANYmal-C):
python scripts/fetch_assets.py

python train.py --list                  # the task catalog
python train.py --task Go2              # headless train (rl_games)
python train.py --task Ant --view       # windowed: watch it train
python play.py  --task Go2 --checkpoint runs/Go2_.../nn/Go2.pth --cmd 1,0,0
```

## Tasks

| Task | Domain | Robot | Description |
|---|---|---|---|
| `Cartpole` | classic | cartpole (USD) | Balance a pole on a force-controlled cart. |
| `Ant` | locomotion | MuJoCo ant (MJCF) | Run forward as fast as possible (torque control). |
| `Anymal` | locomotion | ANYmal-C (URDF) | Velocity-command walking, 11-term flat-terrain reward. |
| `Go2` | locomotion | Unitree Go2 (URDF) | Velocity-command walking, minimal 6-term reward. |
| `Go1` | locomotion | Unitree Go1 (MJCF) | Velocity-command walking. |
| `A1` | locomotion | Unitree A1 (MJCF) | Velocity-command walking. |
| `H1` | locomotion | Unitree H1 (MJCF) | Bipedal velocity walking (19 DOF humanoid). |
| `FrankaReach` | manipulation | Franka Panda (MJCF) | End-effector reach to a random 3D target. |
| `FrankaLift` | manipulation | Franka Panda + gripper | Grasp a cube and hold it at a goal height. |
| `FrankaCabinet` | manipulation | Franka + sektion cabinet | Open a drawer (multi-articulation env). |
| `AllegroCube` | hands | Wonik Allegro Hand | In-hand cube reorientation to a goal pose. |

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

## Testing

```bash
python -m pytest -q                            # Tier 0: engine-free consistency (~2s)
python scripts/run_engine_tests.py --smoke     # Tier 1: per-robot ingest smoke (GPU)
python scripts/run_engine_tests.py --train     # Tier 2: mini-train regression (GPU)
python scripts/preflight.py                    # the pre-push ritual (~90s, all of the above)
python tests/soak/soak_task.py --task FrankaLift --hours 12   # endurance, before milestones
python tools/bench_steps.py --task Ant --envs 256,1024,4096   # perf instrument
```

Tier 0 needs no engine and runs in CI (`.github/workflows/tier0.yml`). Engine
tests run **one world per process** (driven by `scripts/run_engine_tests.py`,
never by a shared pytest session).

## Layout

```
train.py  play.py            unified CLI (registry-driven)
lumengine_envs/              the package: task registry + typed configs
tasks/                       task implementations + _bootstrap
assets/                      robot sources (URDF/MJCF) + per-robot LICENSE; USD is a build artifact
configs/                     optional per-task yaml overrides
tests/                       tier0 (engine-free) / tier1 (smoke) / tier2 (train) / soak
scripts/                     fetch_assets, run_engine_tests, preflight
tools/                       bench_steps + archived bring-up probes
runs/                        trainer checkpoints (gitignored)
```

## Assets & licensing

lm-envs is **Apache-2.0** (see [`LICENSE`](LICENSE)). Robot models keep their
**own** licenses — every redistributed asset is listed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) with its license, origin and
status, and keeps its original license file under `assets/<robot>/`.

Models come exclusively from their **manufacturers' or maintainers' public
URDF/MJCF descriptions** and are converted to USD locally; no proprietary
simulator content is redistributed. Permissively-licensed robots (BSD/MIT/Apache)
are vendored; the rest are fetched on demand by `scripts/fetch_assets.py` —
ANYmal-C is fetched (pinned upstream commit), not vendored.
