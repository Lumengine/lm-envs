"""GPU contact-buffer sizing (the IsaacGym SimParams.physx analogue): SimConfig can raise
the PhysX maxRigidContactCount / foundLostPairsCapacity to avoid contact-pair OVERFLOW at
large env counts. The multiplier scales the PhysX defaults; create_world applies it via the
engine ISettings before the PxScene is built. Verifies the setting is applied and the world
builds and steps with the bumped buffer.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_gpu_buffers.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

NUM_ENVS = 16
MULT = 4
PATH_CONTACTS = "physics/gpu/memory/maxRigidContactCount"
PATH_FOUNDLOST = "physics/gpu/memory/foundLostPairsCapacity"


def run():
    import cartpole_task as C
    import lm.rl as rl
    import lm.core as core
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    cfg = rl.SimConfig(substeps=2, device="auto", gpu_contact_buffer_multiplier=float(MULT))
    task = C.CartpoleTask(num_envs=NUM_ENVS, headless=True, config=cfg)
    sim = task.sim

    # The Python side wrote the bumped values into the engine settings (which PhysXSystem
    # read when it built the PxScene).
    settings = core.get_settings()
    got_c = settings.get_as_int(PATH_CONTACTS)
    got_fl = settings.get_as_int(PATH_FOUNDLOST)
    # Expected = the facade's PhysX-default constants x MULT — import them instead of
    # copying values (the deliberate 256K->512K foundLost bump for terrain scenes
    # already went stale here once).
    from lm.rl import _view as _lmview
    assert got_c == MULT * _lmview._PHYSX_DEFAULT_CONTACTS, f"maxRigidContactCount={got_c}"
    assert got_fl == MULT * _lmview._PHYSX_DEFAULT_FOUNDLOST, f"foundLostPairsCapacity={got_fl}"
    print(f"[test] settings applied: maxRigidContactCount={got_c} foundLostPairsCapacity={got_fl}")

    # The scene built and steps with the enlarged buffer (a too-small buffer would error/overflow).
    for _ in range(4000):
        task.warmup_step(); task.runner.run()
        if task.ready:
            break
    assert task.ready, "articulations did not build"
    for _ in range(10):
        sim.simulate(); sim.fetch_results()
    print(f"[test] world built + stepped with {MULT}x GPU contact buffer (num_envs={NUM_ENVS})")

    print("[test] GPU CONTACT-BUFFER SIZING OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_gpu_contact_buffer_sizing():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except BaseException:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
