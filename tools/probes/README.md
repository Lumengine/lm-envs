# Bring-up diagnostic probes (archive)

One-off diagnostic scripts written during robot bring-up (ant torque probing,
anymal ground-contact debugging, the H1 biped settle/collider/termination
investigations). Nothing imports them; they are kept as executable debugging
memory until the robot-onboarding playbook captures their lessons as docs.

They were written to live in `tasks/` — after the move their `sys.path`
bootstrap points back at the repo root and `tasks/`, so they still run from
here, but expect them to bit-rot: they poke private facade APIs on purpose.

Not part of any package or test tier. Delete freely once superseded.
