"""Large-N instancing: author_world(instanceable=True) shares ONE composed robot prototype
across all envs (huge scenegraph-memory reduction + faster ingest at scale). The physics
ingest must still report a distinct articulation per env (via instance-proxy paths), each at
its own grid cell.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_instanceable.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
import anymal_task as A

ROBOT = _bootstrap.ASSETS / "anymal_converted" / "anymal.usda"
WORLD = _bootstrap.ASSETS / "world_instanceable_test.usd"
NUM_ENVS = 64


def run():
    import torch
    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    rl.author_world(ROBOT, WORLD, num_envs=NUM_ENVS, spacing=A.ENV_SPACING,
                    ground=True, ground_z=A.GROUND_Z, spawn_z=0.0, instanceable=True)

    # The authored stage shares one env prototype (memory win): all env_i are USD instances
    # of a single prototype.
    from pxr import Usd
    stage = Usd.Stage.Open(str(WORLD))
    env_protos = set()
    n_instances = 0
    for i in range(NUM_ENVS):
        p = stage.GetPrimAtPath(f"/World/env_{i}")
        assert p and p.IsInstance(), f"env_{i} is not an instance"
        env_protos.add(p.GetPrototype().GetPath().pathString)
        n_instances += 1
    composed_prims = sum(1 for _ in stage.Traverse())   # prototype NOT expanded into main tree
    assert len(env_protos) == 1, f"envs should share ONE prototype, got {len(env_protos)}"
    print(f"[test] {n_instances} envs share 1 prototype; main scenegraph={composed_prims} prims "
          f"(vs ~{NUM_ENVS * 260} if expanded)")
    del stage

    sim, runner = rl.create_world(str(WORLD), num_envs=NUM_ENVS, dofs_per_actor=A.N_DOF,
                                  config=rl.SimConfig(substeps=2, device="auto",
                                                      gpu_contact_buffer_multiplier=2.0),
                                  headless=True, title="instanceable")
    sim.play()
    for k in range(15000):
        sim.simulate(); sim.fetch_results(); runner.run()
        if sim._batch_ready():
            break
    assert sim._batch_ready(), "batch never became ready (instanceable ingest failed)"

    dof = sim.acquire_dof_state_tensor(); sim.refresh_dof_state_tensor()
    root = sim.acquire_root_state_tensor(); sim.refresh_root_state_tensor()
    assert tuple(dof.shape) == (NUM_ENVS, A.N_DOF, 2), f"dof shape {tuple(dof.shape)}"
    # Each instance built its OWN articulation at its OWN cell — distinct (x,y) per env.
    xy = root[:, 0:2]
    distinct = len(set((round(float(x), 1), round(float(y), 1)) for x, y in xy))
    print(f"[test] ingest built {NUM_ENVS} articulations (dof {tuple(dof.shape)}); "
          f"{distinct} distinct env positions")
    assert distinct >= NUM_ENVS - 1, f"envs not at distinct cells: {distinct}/{NUM_ENVS}"
    print("[test] INSTANCEABLE LARGE-N OK")

    rl.destroy_world(sim, runner)
    return 0


def test_instanceable_large_n():
    assert run() == 0


if __name__ == "__main__":
    try:
        _code = run()
    except BaseException:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
    os._exit(_code)
