"""Test the asymmetric actor-critic (privileged value) path: rl_games 1.6.5 enables a
central value network from central_value_config, reads the privileged state from the obs
DICT (obs["states"]), and the env must advertise state_space. Uses a cartpole with extra
privileged state.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_asymmetric.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 64
NUM_STATES = 6   # 4 obs + 2 privileged (dof velocities)


def run():
    import lumengine_envs.tasks.cartpole_task as C
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    class AsymCartpole(C.CartpoleTask):
        def _compute_states(self):
            self.states_buf[:, :4] = self.obs_buf
            self.states_buf[:, 4:6] = self._dof[:, :, 1]   # privileged: raw dof velocities

    task = AsymCartpole(num_envs=NUM_ENVS, headless=True, num_states=NUM_STATES)
    assert task.num_states == NUM_STATES and task.states_buf is not None

    from lm.rl._rlgames import RlGamesEnv, _warmup, build_default_config
    _warmup(task)

    # Adapter plumbing: state_space advertised, obs returned as {"obs","states"}.
    env = RlGamesEnv(task)
    info = env.get_env_info()
    assert "state_space" in info and tuple(info["state_space"].shape) == (NUM_STATES,), info.get("state_space")
    o = env.reset()
    assert isinstance(o, dict) and set(o) == {"obs", "states"}, type(o)
    assert tuple(o["obs"].shape) == (NUM_ENVS, 4) and tuple(o["states"].shape) == (NUM_ENVS, NUM_STATES)
    o2, _, _, _ = env.step(torch.zeros(NUM_ENVS, 1, device=task.device))
    assert isinstance(o2, dict) and tuple(o2["states"].shape) == (NUM_ENVS, NUM_STATES)
    print(f"[test] adapter asymmetric plumbing OK (state_space={NUM_STATES}, obs is dict {{obs,states}})")

    # Config carries central_value_config when num_states>0.
    cfg = build_default_config("Asym", "e", NUM_ENVS, 4, 1, num_states=NUM_STATES)
    cv = cfg["params"]["config"].get("central_value_config")
    assert cv is not None and cv["network"].get("central_value") is True, cv
    print("[test] build_default_config emits central_value_config (central_value=True)")

    # Integration: a short train must run end-to-end with the central value enabled
    # (a wrong state_space / obs-dict / cv-config would crash rl_games here).
    rl.train_rl_games(task, max_epochs=3, seed=0)
    print("[test] short asymmetric train completed (central value consumed)")

    print("[test] ASYMMETRIC ACTOR-CRITIC OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_asymmetric_actor_critic():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
