"""Phase 1 reliability test: `rl.destroy_world` is a REAL teardown, not os._exit.

Builds a small cartpole world headless, steps it, tears it down with
rl.destroy_world(sim, runner) — which must now RETURN (default contract) with
the framework verified down — then proves the interpreter is still alive
(atexit + post-teardown Python work) and exits 0 NORMALLY via sys.exit. The
pass criterion is the process surviving BOTH the teardown AND the normal
interpreter exit (historically, exiting with a live/partially-torn world
segfaulted in static/interpreter teardown, which is why destroy_world used to
hard-exit).

    set LUMENGINE_ROOT=...\\Lumengine-rl & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_teardown.py
    echo %errorlevel%        (must be 0, printed AFTER the atexit line)

Optional (EXPERIMENTAL, not part of the base pass criterion): set
LM_RL_TEARDOWN_RECREATE=1 to build + destroy a SECOND world in the same
process after the first teardown.
"""
import atexit
import os
import sys
from pathlib import Path
from lumotion_envs._engine import ensure_engine
ensure_engine()
from lumotion_envs import assets as _assets
import lm.rl as rl

ROBOT = _assets.ASSETS / "cartpole_converted" / "cartpole.usda"
WORLD = _assets.ASSETS / "world_teardown_test.usd"
NUM_ENVS = 2
NUM_DOFS = 2   # cartpole: cart (prismatic) + pole (revolute)


def _build_and_step(tag):
    """Author + open a small cartpole world headless, step it until the batch is
    ready plus a few tensor-traffic steps. Returns (sim, runner)."""
    rl.author_world(ROBOT, WORLD, num_envs=NUM_ENVS, spacing=4.0)
    sim, runner = rl.create_world(str(WORLD), num_envs=NUM_ENVS, dofs_per_actor=NUM_DOFS,
                                  config=rl.SimConfig(substeps=2, device="auto"),
                                  headless=True, title=f"Teardown test ({tag})")
    sim.play()
    for _ in range(4000):
        sim.simulate(); sim.fetch_results(); runner.run()
        if sim._batch_ready():
            break
    assert sim._batch_ready(), f"[{tag}] batch never became ready"

    # Real GPU traffic before teardown: acquire + refresh a tensor, then a few
    # more steps, so the drain in destroy_world has something to drain.
    dof = sim.acquire_dof_state_tensor()
    for _ in range(5):
        sim.simulate(); sim.fetch_results()
    sim.refresh_dof_state_tensor()
    assert tuple(dof.shape) == (NUM_ENVS, NUM_DOFS, 2), dof.shape
    print(f"[test] [{tag}] world up and stepping (dof shape {tuple(dof.shape)})")
    return sim, runner


def run():
    import torch
    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available (direct-GPU batch required)")
        return 0

    # atexit was DEAD under the old os._exit teardown — its firing at normal
    # interpreter exit is part of what this test certifies.
    atexit.register(lambda: print("[test] atexit hook ran (interpreter exited "
                                  "normally after teardown)", flush=True))

    sim, runner = _build_and_step("world 1")

    survived = rl.destroy_world(sim, runner)   # default: full teardown, NO process exit
    assert survived is True, f"destroy_world did not report a verified teardown: {survived}"

    import lm.bootstrap as bootstrap
    assert not bootstrap.is_initialized(), "framework still initialized after destroy_world"

    # The interpreter must be fully functional after teardown: allocate, gc,
    # touch torch (its CUDA context outlives the engine by design).
    import gc
    gc.collect()
    t = torch.ones(1024, device="cuda" if torch.cuda.is_available() else "cpu")
    assert float(t.sum()) == 1024.0
    print("[test] interpreter alive after destroy_world (framework down, torch OK)")

    # EXPERIMENTAL second world in the same process — opt-in so the base test
    # stays deterministic while re-entrant bring-up is not a supported contract.
    if os.environ.get("LM_RL_TEARDOWN_RECREATE") == "1":
        sim2, runner2 = _build_and_step("world 2")
        assert rl.destroy_world(sim2, runner2) is True, "second destroy_world failed"
        assert not bootstrap.is_initialized()
        print("[test] RECREATE OK: second world built, stepped and destroyed")

    print("[test] TEARDOWN re-entrant destroy_world OK — exiting NORMALLY (sys.exit)")
    return 0


def test_teardown_reentrant():
    """pytest entry point (requires the lm.rl runtime env + CUDA)."""
    assert run() == 0


if __name__ == "__main__":
    # NB: unlike the historical sim tests (which hard-exited inside destroy_world),
    # this test RETURNS and exits normally — so let SystemExit propagate; catching
    # BaseException would turn the clean sys.exit(0) into a false failure.
    try:
        sys.exit(run())      # NORMAL exit on purpose — surviving it IS the test
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
