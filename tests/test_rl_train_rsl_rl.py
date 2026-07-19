"""Smoke test: the same VecTask trains end-to-end on rsl_rl (the locomotion-standard
on-policy trainer) via rl.train_rsl_rl, not just rl_games.

A wrong VecEnv shape (missing episode_length_buf, the extras["observations"] dict, or
the time_outs pass-through) would crash rsl_rl's OnPolicyRunner here. Runs a couple of
PPO iterations on a small cartpole.

SKIPs cleanly when CUDA is absent or rsl-rl-lib is not installed.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    pip install "rsl-rl-lib==2.2.4"
    python tests/test_rl_train_rsl_rl.py
"""
import os
import sys
import tempfile
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
        import rsl_rl  # noqa: F401
    except ImportError:
        print("[test] SKIP: rsl-rl-lib not installed (pip install rsl-rl-lib==2.2.4)")
        return 0

    task = C.CartpoleTask(num_envs=NUM_ENVS, headless=True)

    # Adapter plumbing: the duck-typed VecEnv exposes what rsl_rl reads.
    from lm.rl._rsl_rl import RslRlEnv
    env = RslRlEnv(task)
    from lm.rl._trainer_util import warmup as _warmup
    _warmup(task)
    obs, extras = env.get_observations()
    assert tuple(obs.shape) == (NUM_ENVS, task.num_obs), obs.shape
    assert "observations" in extras, extras            # rsl_rl indexes this unconditionally
    assert env.num_actions == task.num_actions and env.num_envs == NUM_ENVS
    assert env.episode_length_buf is task.progress_buf
    print("[test] rsl_rl VecEnv plumbing OK (obs/extras/episode_length_buf)")

    # Integration: a couple of PPO iterations must run end-to-end.
    log_dir = tempfile.mkdtemp(prefix="lm_rsl_smoke_")
    runner = rl.train_rsl_rl(task, max_iterations=2, seed=0, log_dir=log_dir)
    assert runner is not None and runner.alg is not None
    print("[test] rsl_rl 2-iteration train completed")

    print("[test] RSL_RL SMOKE OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_train_rsl_rl():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
