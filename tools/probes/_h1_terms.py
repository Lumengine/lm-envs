"""Why does every H1 env terminate on step 1 (fails~=num_envs, ep_len=0)? Build the
task, warm up + _capture (real settle + managers), reset to home, then print per-link
contact force and evaluate EACH termination term at the home pose. Whatever fires at
the freshly-reset home pose is the poison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _bootstrap
_bootstrap.bootstrap()
import lm.rl as rl
import torch
from lumengine_envs.config import H1Config
from legged_velocity import LeggedVelocityTask


def main():
    task = LeggedVelocityTask(cfg=H1Config(num_envs=16, headless=True))
    for _ in range(400):
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        if task.ready:
            break
    task._capture(); task._captured = True

    # reset all envs to home, then advance and probe contact at 1 vs 10 steps: is the
    # knee contact a first-frame-post-reset transient (foot not yet planted) or persistent?
    ids = torch.arange(task.num_envs, device=task.device)
    task._reset_idx(ids)
    for step in range(1, 11):
        task.sim.set_dof_position_target_tensor(task._default_dof)
        task.sim.simulate(); task.sim.fetch_results()
        if task.runner is not None:
            task.runner.run()
        task._update_state()
        if step in (1, 3, 10):
            kf = task._contact[:, task.knee_indices, :].norm(dim=2).mean(0)
            ff = task._contact[:, task.feet_indices, :].norm(dim=2).mean(0)
            print(f"  post-reset step {step:2}: base z={float(task._root[:,2].mean()):+.3f} "
                  f"knee F={[round(float(x),1) for x in kf]} foot F={[round(float(x),1) for x in ff]} "
                  f"resets={int(task.terminations.compute().sum())}/{task.num_envs}")

    lm = task.robot.view.link_map
    inv = {i: n for n, i in lm.items()}
    contact = task._contact
    fmag = contact.norm(dim=2).mean(dim=0)
    print(f"\nhome base z={float(task._root[:,2].mean()):+.3f}  up_proj={float(task._up_proj.mean()):.3f}")
    print("per-link mean |contact force| (N), >0.5 flagged:")
    for i in range(fmag.shape[0]):
        flag = "*" if float(fmag[i]) > 0.5 else " "
        print(f"  {flag} link[{i:2}] {inv.get(i,'?'):26} = {float(fmag[i]):8.2f} N")

    # Evaluate each termination term in isolation.
    print("\ntermination link picks:")
    print("  base link (0) :", inv.get(0))
    print("  knee_indices  :", [inv.get(int(i)) for i in task.knee_indices])
    base_f = contact[:, 0, :].norm(dim=1)
    knee_f = contact[:, task.knee_indices, :].norm(dim=2)
    print(f"\n  base_contact:  link0 F(mean)={float(base_f.mean()):.2f}  "
          f"envs>1N={int((base_f>1.0).sum())}/{task.num_envs}")
    print(f"  knee_contact:  knee F(mean)={[round(float(x),2) for x in knee_f.mean(0)]}  "
          f"envs any>1N={int((knee_f>1.0).any(dim=1).sum())}/{task.num_envs}")
    print(f"  tipped:        up_proj(mean)={float(task._up_proj.mean()):.2f}  "
          f"envs<{task.cfg.upright_min}={int((task._up_proj<task.cfg.upright_min).sum())}/{task.num_envs}")
    total = task.terminations.compute()
    print(f"\n  TOTAL reset_buf = {int(total.sum())}/{task.num_envs}")
    rl.destroy_world(task.sim, task.runner)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback; traceback.print_exc()
