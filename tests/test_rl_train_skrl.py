"""Smoke test: the same VecTask trains end-to-end on skrl (broad PyTorch algorithm
coverage) via rl.train_skrl, not just rl_games.

Exercises the full skrl path: SkrlEnv -> wrap_env(..., "isaacgym-preview4") -> default
PPO + SequentialTrainer. A wrong space construction, the 4-tuple step contract, or the
runner-tick would crash skrl here. Runs a few hundred timesteps on a small cartpole.

SKIPs cleanly when CUDA is absent or skrl is not installed.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    pip install "skrl==1.4.3"
    python tests/test_rl_train_skrl.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 16


def run():
    import lumotion_envs.tasks.cartpole_task as C
    import lumotion as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0
    try:
        import skrl  # noqa: F401
    except ImportError:
        print("[test] SKIP: skrl not installed (pip install skrl==1.4.3)")
        return 0

    task = C.CartpoleTask(num_envs=NUM_ENVS, headless=True)

    # Low-level surface: wrap_skrl warms up + returns a wrapped env with skrl spaces.
    import gymnasium
    env = rl.wrap_skrl(task)
    assert env.num_envs == NUM_ENVS
    assert isinstance(env.observation_space, gymnasium.spaces.Box)
    assert env.action_space.shape == (task.num_actions,)
    print("[test] skrl wrap_skrl OK (wrapped env + spaces)")

    # Integration: a short PPO run must complete through SequentialTrainer.
    agent = rl.train_skrl(task, timesteps=256, rollouts=16, seed=0, headless=True)
    assert agent is not None
    print("[test] skrl 256-timestep train completed")

    print("[test] SKRL SMOKE OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_train_skrl():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
