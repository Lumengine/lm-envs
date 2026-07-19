"""Phase 3 test: per-articulation-link net contact force (legged foot contact).

A standing anymal is settled on the ground under its PD drives, then the per-link
net contact force is read. Physical sanity: the total upward (z) contact force over
all links per env must be positive and ~ balance the robot's weight (it is standing).

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_link_contact.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 4
SETTLE_STEPS = 200


def run():
    import lumotion_envs.tasks.anymal_task as A
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available (contact-force kernel is CUDA-only)")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    sim = task.sim
    for _ in range(4000):
        task.warmup_step(); task.runner.run()
        if task.ready:
            break
    assert task.ready, "batch never became ready"

    # Let the robots settle on the ground under their PD-held stance.
    for _ in range(SETTLE_STEPS):
        sim.simulate(); sim.fetch_results()

    contact = sim.acquire_link_net_contact_force_tensor()
    assert contact is not None, "link contact tensor is None (batch not ready?)"
    sim.simulate(); sim.fetch_results()
    sim.refresh_link_net_contact_force_tensor()

    assert torch.isfinite(contact).all(), "link contact force has non-finite values"
    n_links = contact.shape[1]
    assert tuple(contact.shape) == (NUM_ENVS, n_links, 3)

    # Per-env total upward contact force (sum over links).
    sum_z = contact[:, :, 2].sum(dim=1)
    total_mag = contact.norm(dim=2).sum(dim=1)
    print(f"[test] per-env sum_z (N)   = {[round(float(v), 1) for v in sum_z]}")
    print(f"[test] per-env |F| sum (N) = {[round(float(v), 1) for v in total_mag]}")

    # Standing robot: net upward contact must be positive and significant (anymal_c is
    # tens of kg -> a few hundred N). Loose bounds: just prove the feet are loaded and
    # the kernel scattered real forces (not all zero, correct sign).
    assert (sum_z > 50.0).all(), f"upward contact force too small / wrong sign: {sum_z}"
    assert (sum_z < 5000.0).all(), f"upward contact force implausibly large: {sum_z}"

    # The load should be concentrated in a few links (the 4 feet), not spread over all 60.
    loaded = (contact.norm(dim=2) > 1.0).sum(dim=1)   # links with >1N contact, per env
    print(f"[test] loaded links per env = {[int(v) for v in loaded]}")
    assert (loaded >= 1).all() and (loaded <= 12).all(), f"unexpected #loaded links: {loaded}"

    print("[test] PHASE3 link net contact force OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_phase3_link_contact():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
