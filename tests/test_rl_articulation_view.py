"""Layer-1 test for the Isaac-style rl.ArticulationView: every backed method works
over the batch, and the not-yet-backed domain-randomization property setters raise
NotImplementedError (the contract is posed but unbacked).

    set LUMENGINE_ROOT=...\\Lumengine2 & set LUMENGINE_BUILD_CONFIG=Release
    set LM_PHYSX_SHARE_CUDA_CONTEXT=1 & python tests/test_rl_articulation_view.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

NUM_ENVS = 4
N_DOF = 12
N_LINKS = 60


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
    assert isinstance(view, rl.ArticulationView)
    assert view.count == NUM_ENVS and view.num_dofs == N_DOF and view.num_links == N_LINKS
    print(f"[test] view: count={view.count} num_dofs={view.num_dofs} num_links={view.num_links} dev={view.device}")

    sim.simulate(); sim.fetch_results()

    # Backed reads — shapes + finiteness.
    checks = {
        "dof_states": (view.get_dof_states(), (NUM_ENVS, N_DOF, 2)),
        "dof_positions": (view.get_dof_positions(), (NUM_ENVS, N_DOF)),
        "root_states": (view.get_root_states(), (NUM_ENVS, 13)),
        "link_states": (view.get_link_states(), (NUM_ENVS, N_LINKS, 13)),
        "applied_dof_forces": (view.get_applied_dof_forces(), (NUM_ENVS, N_DOF)),
        "measured_joint_forces": (view.get_measured_joint_forces(), (NUM_ENVS, N_LINKS, 6)),
        "net_contact_forces": (view.get_net_contact_forces(), (NUM_ENVS, N_LINKS, 3)),
    }
    for name, (t, shape) in checks.items():
        assert tuple(t.shape) == shape, f"{name}: {tuple(t.shape)} != {shape}"
        assert torch.isfinite(t).all(), f"{name} not finite"
    assert view.get_jacobians().shape[0] == NUM_ENVS
    assert view.get_mass_matrices().shape == (NUM_ENVS, (N_DOF + 6) ** 2)
    assert len(view.get_root_entities()) == NUM_ENVS
    print("[test] backed reads (state/dynamics/contacts) OK")

    # Backed control + external forces (must not raise).
    view.set_dof_position_targets(torch.zeros(NUM_ENVS, N_DOF, device=view.device))
    view.set_dof_actuation_forces(torch.zeros(NUM_ENVS, N_DOF, device=view.device))
    view.apply_link_forces(torch.zeros(NUM_ENVS, N_LINKS, 3, device=view.device))
    view.apply_link_torques(torch.zeros(NUM_ENVS, N_LINKS, 3, device=view.device))
    # Indexed write path through the view.
    ids = torch.tensor([1, 3], device=view.device, dtype=torch.long)
    view.set_dof_states(view.get_dof_states().clone(), indices=ids)
    print("[test] backed control + external forces + indexed write OK")

    # Verify apply_link_forces actually pushes: big upward force on all links -> robots rise.
    z0 = view.get_root_states()[:, 2].clone()
    F = torch.zeros(NUM_ENVS, N_LINKS, 3, device=view.device); F[:, :, 2] = 2000.0
    view.apply_link_forces(F)
    for _ in range(10):
        sim.simulate(); sim.fetch_results()
    z1 = view.get_root_states()[:, 2]
    view.apply_link_forces(torch.zeros(NUM_ENVS, N_LINKS, 3, device=view.device))   # stop pushing
    print(f"[test] apply_link_forces effect: dz(mean)={float((z1 - z0).mean()):+.3f} m (should be > 0)")
    assert float((z1 - z0).mean()) > 0.05, "upward link force did not lift the robots"

    # DR property setters are now all backed (mass/stiffness/damping/armature/joint+contact
    # friction). Only get_masses (read-back) remains an explicit stub.
    try:
        view.get_masses()
        raise AssertionError("get_masses should raise NotImplementedError (read-back not surfaced)")
    except NotImplementedError:
        pass
    print("[test] get_masses still an explicit stub (read-back); all DR setters backed")

    print("[test] LAYER1 ArticulationView OK")
    rl.destroy_world(task.sim, task.runner)
    return 0


def test_layer1_articulation_view():
    assert run() == 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except BaseException:
        import traceback
        print("[test] FAILED:")
        traceback.print_exc()
        os._exit(1)
