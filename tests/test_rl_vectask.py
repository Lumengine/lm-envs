"""Phase D test: the rl.VecTask base contract, via the migrated CartpoleTask.

Checks the shared loop the base provides: reset() returns clipped obs of the right
shape, step() returns (obs, rew, reset, extras) with correct shapes/finiteness and a
working time_outs truncation, progress advances, and a full episode triggers timeouts.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_vectask.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 8


def run():
    import lumotion_envs.tasks.cartpole_task as C
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    task = C.CartpoleTask(num_envs=NUM_ENVS, headless=True)
    assert isinstance(task, rl.VecTask), "CartpoleTask should derive from rl.VecTask"

    from lm.rl._rlgames import _warmup
    _warmup(task)
    assert task.ready

    # reset() -> clipped obs (num_envs, num_obs)
    obs = task.reset()
    assert tuple(obs.shape) == (NUM_ENVS, task.num_obs), obs.shape
    assert torch.isfinite(obs).all() and float(obs.abs().max()) <= task.clip_obs + 1e-4

    # step() contract + buffers
    for _ in range(5):
        a = torch.zeros(NUM_ENVS, task.num_actions, device=task.device)
        obs, rew, reset, extras = task.step(a)
        assert tuple(obs.shape) == (NUM_ENVS, task.num_obs)
        assert tuple(rew.shape) == (NUM_ENVS,)
        assert tuple(reset.shape) == (NUM_ENVS,)
        assert "time_outs" in extras and tuple(extras["time_outs"].shape) == (NUM_ENVS,)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert float(obs.abs().max()) <= task.clip_obs + 1e-4   # base clamps obs
    assert float(task.progress_buf.max()) >= 5, "progress_buf not advancing"
    print("[test] VecTask step/reset contract + buffers  OK")

    # Time-limit truncation mechanism: with progress at the limit, the next step must
    # flag every env as timed-out (decoupled from cartpole's physics, which would fail
    # under a zero action long before the limit).
    task.reset()
    task.progress_buf[:] = task.max_episode_length - 1
    a = torch.zeros(NUM_ENVS, task.num_actions, device=task.device)
    _, _, reset, extras = task.step(a)
    assert float(extras["time_outs"].sum()) == NUM_ENVS, \
        f"timeout did not fire at the limit: {extras['time_outs']}"
    assert (reset > 0).all(), "envs at the time limit should reset"
    print("[test] VecTask time-limit truncation (time_outs) fires  OK")

    print("[test] PHASE_D VecTask base OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_phase_d_vectask():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
