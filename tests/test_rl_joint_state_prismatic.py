"""UsdPhysics JointStateAPI on a PRISMATIC articulation joint: the authored initial linear
position is honored by the USD->PhysX ingest (mirror of the revolute case).

Authors `state:linear:physics:position = 0.4` (meters) on the cartpole cart slider joint,
ingests, and checks the cart DOF starts at ~0.4 (not 0). Pure ingest check.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_joint_state_prismatic.py
"""
import os
import sys
from pathlib import Path
from lumotion_envs._engine import ensure_engine
ensure_engine()
from lumotion_envs import assets as _assets
import lumotion as rl

ROBOT = _assets.ASSETS / "cartpole_converted" / "cartpole.usda"
WORLD = _assets.ASSETS / "world_jointstate_prismatic_test.usd"
CART_POS = 0.4   # meters
NUM_ENVS = 2
CART_DOF = 0   # cartpole DOF order: 0 = cart (prismatic), 1 = pole (revolute)


def run():
    import torch
    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    rl.author_world(ROBOT, WORLD, num_envs=NUM_ENVS, spacing=4.0)

    from pxr import Usd, UsdPhysics, Sdf
    stage = Usd.Stage.Open(str(WORLD))
    n_prismatic = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.PrismaticJoint):
            prim.CreateAttribute("state:linear:physics:position",
                                 Sdf.ValueTypeNames.Float).Set(CART_POS)
            n_prismatic += 1
    stage.GetRootLayer().Save()
    assert n_prismatic == NUM_ENVS, f"expected {NUM_ENVS} prismatic joints, found {n_prismatic}"
    print(f"[test] authored state:linear:physics:position={CART_POS} on {n_prismatic} cart joints")

    sim, runner = rl.create_world(str(WORLD), num_envs=NUM_ENVS, dofs_per_actor=2,
                                  config=rl.SimConfig(substeps=2, device="auto"),
                                  headless=True, title="JointState prismatic ingest test")
    sim.play()
    for _ in range(4000):
        sim.simulate(); sim.fetch_results(); runner.run()
        if sim._batch_ready():
            break
    assert sim._batch_ready(), "batch never became ready"

    dof = sim.acquire_dof_state_tensor()
    sim.refresh_dof_state_tensor()
    cart = dof[:, CART_DOF, 0]
    print(f"[test] cart DOF after ingest = {[round(float(v), 3) for v in cart]} "
          f"(expected ~{CART_POS}; without the fix it would be ~0)")

    # The authored 0.4 m must show up (the cart holds along its horizontal slider; gravity is
    # off-axis). Band clearly excludes the old 0.
    assert (cart > 0.30).all() and (cart < 0.50).all(), \
        f"prismatic JointStateAPI initial position not honored: {cart}"
    print("[test] PRISMATIC JointStateAPI initial joint position OK")

    rl.destroy_world(sim, runner)
    return 0


def test_prismatic_joint_state_ingest():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
