"""SPIKE + feature test for actor-property DR (the ArticulationView stubs being filled):
do live CPU setters (mass / drive stiffness/damping/armature) affect an already-added
direct-GPU articulation? Definitive mass spike: airborne robots under a fixed upward
link force — nominal-mass rises, heavy-mass falls (sign flip) iff setMass takes effect.

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_actor_props.py
"""
import os
import sys
from pathlib import Path

NUM_ENVS = 4
N_DOF = 12
N_LINKS = 60


def run():
    import lumengine_envs.tasks.anymal_task as A
    import lm.rl as rl
    import torch

    if not torch.cuda.is_available():
        print("[test] SKIP: CUDA not available")
        return 0

    task = A.AnymalTask(num_envs=NUM_ENVS, headless=True)
    for _ in range(4000):
        task.warmup_step(); task.runner.run()
        if task.ready:
            break
    assert task.ready
    view = task.sim.articulations
    sim = task.sim
    dev = view.device

    def airborne_rise(force_per_link_z, steps=8):
        # Teleport high up, zero velocity (no ground contact -> pure F = ma + gravity).
        root = view.get_root_states().clone()
        root[:, 0:2] = 0.0; root[:, 2] = 8.0
        root[:, 3:7] = torch.tensor([0., 0., 0., 1.], device=dev)
        root[:, 7:13] = 0.0
        view.set_root_states(root)
        sim.simulate(); sim.fetch_results()
        z0 = view.get_root_states()[:, 2].clone()
        F = torch.zeros(NUM_ENVS, N_LINKS, 3, device=dev); F[:, :, 2] = force_per_link_z
        for _ in range(steps):
            view.apply_link_forces(F)          # re-apply each step (continuous force)
            sim.simulate(); sim.fetch_results()
        z1 = view.get_root_states()[:, 2]
        view.apply_link_forces(torch.zeros(NUM_ENVS, N_LINKS, 3, device=dev))
        return float((z1 - z0).mean())

    # Upward force tuned so a nominal anymal rises but a heavy one (10 kg/link x 60 = 600 kg, ~5.9 kN
    # weight) falls under the same force.
    Fz = 80.0   # per link -> 4800 N total
    rise_nominal = airborne_rise(Fz)
    view.set_masses(torch.full((NUM_ENVS, N_LINKS), 10.0, device=dev))
    rise_heavy = airborne_rise(Fz)
    print(f"[test] MASS SPIKE: rise_nominal={rise_nominal:+.3f} m  rise_heavy={rise_heavy:+.3f} m")
    assert rise_nominal > 0.05, "nominal robot should rise under the upward force"
    assert rise_heavy < rise_nominal - 0.05, "heavier mass did not reduce the rise -> setMass had NO live effect"
    print("[test] -> live CPU setMass DOES affect the direct-GPU sim (spike PASSES)")

    # Drive-param setters: confirm they run + take effect. Set stiffness ~0 (limp) vs high,
    # command a target offset airborne, and check the joints track the target far better with
    # high stiffness (the drive force scales with stiffness).
    def track_error(stiffness, steps=12):
        view.set_dof_stiffnesses(torch.full((NUM_ENVS, N_DOF), float(stiffness), device=dev))
        root = view.get_root_states().clone()
        root[:, 2] = 8.0; root[:, 7:13] = 0.0
        view.set_root_states(root)
        dof = view.get_dof_states()
        target = dof[:, :, 0].clone() + 0.4
        view.set_dof_position_targets(target)
        for _ in range(steps):
            sim.simulate(); sim.fetch_results()
        cur = view.get_dof_states()[:, :, 0]
        return float((cur - target).abs().mean())

    err_low = track_error(2.0)
    err_high = track_error(400.0)
    print(f"[test] STIFFNESS: track_err low(k=2)={err_low:.3f} rad  high(k=400)={err_high:.3f} rad")
    assert err_high < err_low - 0.02, "higher drive stiffness did not improve tracking -> setDriveParams no effect"
    print("[test] -> live CPU setDriveParams (stiffness) DOES affect the sim")

    # Damping + armature: plumbing (run without error on the live articulation).
    view.set_dof_dampings(torch.full((NUM_ENVS, N_DOF), 5.0, device=dev))
    view.set_dof_armatures(torch.full((NUM_ENVS, N_DOF), 0.01, device=dev))
    print("[test] damping + armature setters run OK")

    print("[test] ACTOR-PROPERTY DR (mass/stiffness/damping/armature) OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_actor_property_dr():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
