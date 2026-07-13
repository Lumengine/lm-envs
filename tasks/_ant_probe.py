"""Physics probe: can the MJCF ant actually MOVE under joint torque? No policy — just
apply a fixed torque pattern at several amplitudes and measure joint response + body
displacement. Isolates "torque too weak" from "RL stuck standing".

    set LUMENGINE_ROOT=...  & set LM_PHYSX_SHARE_CUDA_CONTEXT=1
    py -3.11 tasks/_ant_probe.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
import torch

_ANT = _bootstrap.ASSETS / "ant.xml"
_CFG = _bootstrap.ASSETS / "ant.rl.yaml"
N_DOF = 8
N = 16


VIEW = os.environ.get("LM_RL_VIEW") == "1"


def build():
    world = rl.World(num_envs=N, env_spacing=2.5)
    world.add_ground(z=0.0, friction=1.0)
    robot = world.add_robot(rl.Mjcf(str(_ANT), config=str(_CFG)), spawn_z=0.62)
    sim, runner = world.build(headless=not VIEW,
                             config=rl.SimConfig(substeps=2, device="auto",
                                                 gpu_contact_buffer_multiplier=2.0),
                             title="ant probe oscillating torque")
    sim.play()
    # warm up until the batch is ready — headless physics only advances on a runner tick
    # (the app's Physics phase), exactly like the rl_games adapter does each step.
    for _ in range(300):
        sim.simulate(); sim.fetch_results()
        if runner is not None:
            runner.run()
        if sim._batch_ready():
            break
    return world, robot, sim, runner


def step(sim, runner):
    sim.simulate(); sim.fetch_results()
    if runner is not None:
        runner.run()


def main():
    world, robot, sim, runner = build()
    dev = sim.device
    dof = sim.acquire_dof_state_tensor()
    root = sim.acquire_root_state_tensor()
    sim.refresh_dof_state_tensor(); sim.refresh_root_state_tensor()
    print(f"device={dev} num_dofs={sim.num_dofs}")
    print(f"spawn root xyz={[round(float(x),3) for x in root[0,0:3]]}")

    if VIEW:
        # Windowed: drive a continuous oscillating-torque "gait" so you can WATCH the ants
        # flail and translate across the ground — visual proof the base is floating (not
        # anchored). Runs until you close the window.
        try:
            runner.frame(eye=(6.0, -6.0, 5.0), target=(4.0, 4.0, 0.8))
        except Exception:
            pass
        T = float(os.environ.get("LM_RL_ANT_TORQUE", "40.0"))
        t = 0
        while True:
            phase = 1.0 if (t // 20) % 2 == 0 else -1.0
            a = torch.zeros(N, N_DOF, device=dev)
            a[:, 0:N_DOF:2] = phase
            a[:, 1:N_DOF:2] = -phase
            sim.set_dof_actuation_force_tensor(a * T)
            sim.simulate(); sim.fetch_results()
            if runner is not None and runner.run() is False:
                break
            sim.refresh_root_state_tensor()
            t += 1
            if t % 120 == 0:
                print(f"[view] t={t} torso z(mean)={float(root[:,2].mean()):.2f} "
                      f"x-spread={float(root[:,0].max()-root[:,0].min()):.2f}", flush=True)
        rl.destroy_world(sim, runner)
        return

    # PHASE 0 — gravity only (zero torque): does the torso FALL? If the base is anchored
    # (fixed), z stays put; if floating, z changes as the legs splay/settle.
    x0 = root[:, 0].clone(); y0 = root[:, 1].clone(); z0 = root[:, 2].clone()
    for _ in range(120):
        sim.set_dof_actuation_force_tensor(torch.zeros(N, N_DOF, device=dev))
        step(sim, runner); sim.refresh_root_state_tensor()
    print(f"[gravity-only 120 steps] dz={float((root[:,2]-z0).mean()):+.4f}  "
          f"|dxy|={float(((root[:,0]-x0)**2+(root[:,1]-y0)**2).sqrt().mean()):.4f}  "
          f"z_now={float(root[:,2].mean()):.3f}  (anchored if dz~0 and |dxy|~0)")

    # A simple oscillating gait pattern: alternate torque sign across the 4 legs over time.
    for T in (15.0, 40.0, 80.0, 150.0):
        # reset to stance
        sim.refresh_root_state_tensor()
        z0 = root[:, 2].clone()
        x0 = root[:, 0].clone()
        max_dofvel = 0.0
        max_h = 0.0
        for t in range(150):
            phase = 1.0 if (t // 15) % 2 == 0 else -1.0
            a = torch.zeros(N, N_DOF, device=dev)
            a[:, 0:N_DOF:2] = phase        # hips one way
            a[:, 1:N_DOF:2] = -phase       # ankles the other
            sim.set_dof_actuation_force_tensor(a * T)
            step(sim, runner)
            sim.refresh_dof_state_tensor(); sim.refresh_root_state_tensor()
            max_dofvel = max(max_dofvel, float(dof[:, :N_DOF, 1].abs().mean()))
            max_h = max(max_h, float(root[:, 2].mean()))
        dx = float((root[:, 0] - x0).abs().mean())
        dz = float((root[:, 2] - z0).mean())
        print(f"T={T:6.1f} | mean|dof_vel| peak={max_dofvel:6.2f} rad/s "
              f"| body |dx|={dx:5.3f} m | final dz={dz:+.3f} | peak h={max_h:.2f}")

    rl.destroy_world(sim, runner)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc()
