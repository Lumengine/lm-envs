"""Smoke-test the Allegro hand asset: build a world with just the hand (fixed base),
warm up the direct-GPU batch, and verify it materializes as a 16-DOF articulation with
per-link collision. De-risks the asset before writing the reorientation task."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tasks"))
import _bootstrap
_bootstrap.bootstrap()
ASSETS = _bootstrap.ASSETS
import torch  # noqa: E402
import lm.rl as rl  # noqa: E402

world = rl.World(num_envs=4, env_spacing=1.0)
world.add_ground(z=-0.5, friction=1.0)
hand = world.add_robot(rl.Mjcf(str(ASSETS / "allegro/right_hand.xml"), prep=True,
                               config=str(ASSETS / "allegro.rl.yaml")), spawn_z=0.3)
sim, runner = world.build(headless=True, instanceable=False,
                          config=rl.SimConfig(dt=1.0 / 60.0, substeps=2, device="auto"),
                          title="AllegroSmoke")
sim.play()
i = 0
while not (sim._batch_ready() if hasattr(sim, "_batch_ready") else True):
    sim.simulate(); sim.fetch_results()
    if runner is not None:
        runner.run()
    i += 1
    if i > 4000:
        print("never ready"); sys.exit(3)

view = hand.view
print("==== ALLEGRO SMOKE ====")
print(f"batch ready after {i} steps")
print(f"num_dofs (per hand)  : {sim._dofs_per_actor}")
print(f"joint names          : {sorted(view.dof_indices.__self__.link_map) if hasattr(view,'dof_indices') else 'n/a'}")
dof = sim.acquire_dof_state_tensor()
contact = sim.acquire_link_net_contact_force_tensor()
sim.refresh_dof_state_tensor()
print(f"dof state shape      : {tuple(dof.shape)}")
print(f"link contact shape   : {tuple(contact.shape)}  (rows, links, 3)")
# Step a bit under gravity holding the default pose; the hand should stay put (fixed base).
zero = torch.zeros(sim._num_envs, sim._dofs_per_actor, device=dof.device)
for k in range(60):
    sim.set_dof_position_target_tensor(zero) if hasattr(sim, "set_dof_position_target_tensor") else None
    sim.simulate(); sim.fetch_results()
    if runner is not None:
        runner.run()
sim.refresh_dof_state_tensor()
print(f"final |dof pos| max  : {dof[:, :, 0].abs().max().item():.4f} (small = fingers hold default)")
print("VERDICT: asset builds as a 16-DOF articulation" if sim._dofs_per_actor == 16
      else f"VERDICT: unexpected DOF count {sim._dofs_per_actor}")
rl.destroy_world(sim, runner)
