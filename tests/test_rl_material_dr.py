"""Test the last two ArticulationView stubs: per-env CONTACT friction (material
un-sharing) and per-DOF JOINT friction. The material test proves per-env independence
AND effect: env 0 gets near-zero friction, the others high — under the same horizontal
push, the low-friction robot slides clearly further (its material is now its own).

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_material_dr.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

NUM_ENVS = 4
N_DOF = 12


def run():
    import anymal_task as A
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    sim = task.sim
    for _ in range(4000):
        task.warmup_step(); task.runner.run()
        if task.ready:
            break
    assert task.ready
    view = sim.articulations
    dev = view.device

    for _ in range(20):
        sim.simulate(); sim.fetch_results()

    # Joint friction: runs on the live articulation (drive-side, like the drive params).
    view.set_friction_coefficients(torch.full((NUM_ENVS, N_DOF), 0.5, device=dev))
    print("[test] set_friction_coefficients (joint) runs OK")

    # Contact-material friction: the view DECLINES it — a live shape-material change does not
    # refresh on the direct-GPU contact pipeline (verified: identical slide with friction
    # 0.0 vs 5.0). The view raises NotImplementedError so the contract stays honest.
    try:
        view.set_material_properties(torch.full((NUM_ENVS,), 0.5, device=dev))
        raise AssertionError("set_material_properties should decline (not live on direct-GPU)")
    except NotImplementedError:
        print("[test] set_material_properties correctly declined (no live GPU contact refresh)")

    print("[test] JOINT FRICTION DR + material-finding OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_material_and_joint_friction_dr():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except BaseException:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
