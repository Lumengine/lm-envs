"""Phase 2 test: indexed (partial-env) writes — the IsaacGym reset_idx pattern.

Asserts that set_*_tensor_indexed writes ONLY the selected batch rows and leaves
the others bit-unchanged. Uses DOF velocity + root x-position (neither is joint-limit
clamped) so the write->read round-trip is exact, with no simulate in between.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_indexed_reset.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 6


def run():
    import lumotion_envs.tasks.anymal_task as A
    import lumotion as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available (indexed writes are CUDA-only)")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    sim = task.sim
    for _ in range(4000):
        task.warmup_step(); task.runner.run()
        if task.ready:
            break
    assert task.ready, "batch never became ready"
    n, d = sim.num_envs, sim.num_dofs
    dev = sim.device
    ids = torch.tensor([1, 3, 5], device=dev, dtype=torch.long)
    others = torch.tensor([0, 2, 4], device=dev, dtype=torch.long)

    # --- DOF state indexed -------------------------------------------------
    dof = sim.acquire_dof_state_tensor()
    base = torch.zeros(n, d, 2, device=dev)
    base[:, :, 1] = 1.0                      # vel = 1.0 everywhere
    sim.set_dof_state_tensor(base)
    sim.refresh_dof_state_tensor()
    assert torch.allclose(dof[:, :, 1], torch.ones_like(dof[:, :, 1]), atol=1e-4), \
        f"full DOF write failed: {dof[:, :, 1].mean()}"

    sel = base.clone()
    sel[ids, :, 1] = 2.0                     # only rows 1,3,5 -> vel 2.0
    sim.set_dof_state_tensor_indexed(sel, ids)
    sim.refresh_dof_state_tensor()
    assert torch.allclose(dof[ids][:, :, 1], torch.full((ids.numel(), d), 2.0, device=dev), atol=1e-4), \
        f"indexed rows not updated: {dof[ids][:, :, 1]}"
    assert torch.allclose(dof[others][:, :, 1], torch.ones(others.numel(), d, device=dev), atol=1e-4), \
        f"NON-indexed rows were clobbered: {dof[others][:, :, 1]}"
    print("[test] DOF indexed write: selected rows updated, others untouched  OK")

    # --- root state indexed ------------------------------------------------
    root = sim.acquire_root_state_tensor()
    rbase = torch.zeros(n, 13, device=dev)
    rbase[:, 2] = 0.6        # z
    rbase[:, 6] = 1.0        # quat w (identity, xyzw)
    sim.set_root_state_tensor(rbase)
    sim.refresh_root_state_tensor()
    assert torch.allclose(root[:, 0], torch.zeros(n, device=dev), atol=1e-4), "full root write x failed"

    rsel = rbase.clone()
    rsel[ids, 0] = 5.0       # only rows 1,3,5 -> x = 5
    sim.set_root_state_tensor_indexed(rsel, ids)
    sim.refresh_root_state_tensor()
    assert torch.allclose(root[ids][:, 0], torch.full((ids.numel(),), 5.0, device=dev), atol=1e-3), \
        f"indexed root rows not updated: {root[ids][:, 0]}"
    assert torch.allclose(root[others][:, 0], torch.zeros(others.numel(), device=dev), atol=1e-3), \
        f"NON-indexed root rows were clobbered: {root[others][:, 0]}"
    print("[test] root indexed write: selected rows updated, others untouched  OK")

    print("[test] PHASE2 indexed reset OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_phase2_indexed_reset():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
