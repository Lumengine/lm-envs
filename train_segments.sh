#!/usr/bin/env bash
# Segmented training driver — chains rl_games training across the flaky mid-run
# process death (sustained-gripper GPU contact fault). Each segment resumes from the
# newest checkpoint of the previous one; rl_games restores the epoch counter, so
# max_epochs is ABSOLUTE and segments accumulate toward the target. Stops when a
# segment exits cleanly (reached the target) or after MAX_SEG attempts.
#
#   ./train_segments.sh <Task> <target_epochs> [max_segments]
set -u
TASK="${1:?task}"; TARGET="${2:?target epochs}"; MAX_SEG="${3:-12}"
export LUMENGINE_ROOT="C:/Users/perri/Software/Lumengine/Lumengine-rl"
export LM_PHYSX_SHARE_CUDA_CONTEXT=1
cd "$(dirname "$0")"

ckpt="${INIT_CKPT:-}"     # optional: seed segment 1 from an existing checkpoint
for seg in $(seq 1 "$MAX_SEG"); do
  echo "=== SEGMENT $seg/$MAX_SEG  (resume_from='${ckpt:-none}') ==="
  if [ -n "$ckpt" ]; then
    python -u train.py --task "$TASK" --epochs "$TARGET" --resume-from "$ckpt" \
      > "seg_${TASK}_${seg}.log" 2>&1
  else
    python -u train.py --task "$TASK" --epochs "$TARGET" \
      > "seg_${TASK}_${seg}.log" 2>&1
  fi
  code=$?
  # newest .pth by mtime across all runs (the just-created segment's latest checkpoint)
  newest=$(ls -t runs/${TASK}_*/nn/*.pth 2>/dev/null | head -1)
  ep=$(grep -oE "epoch: [0-9]+/" "seg_${TASK}_${seg}.log" | tail -1 | grep -oE "[0-9]+")
  echo "    segment exit=$code  reached_epoch=${ep:-?}  newest_ckpt=${newest:-none}"
  if [ "$code" -eq 0 ]; then
    echo "=== DONE: segment $seg exited cleanly (target reached). ckpt=$newest ==="
    echo "$newest" > "segments_${TASK}_final.txt"
    exit 0
  fi
  if [ -z "$newest" ] || [ "$newest" = "$ckpt" ]; then
    echo "=== STALL: no new checkpoint produced; aborting. ==="
    exit 2
  fi
  ckpt="$newest"
done
echo "=== EXHAUSTED $MAX_SEG segments without clean completion. last ckpt=$ckpt ==="
echo "$ckpt" > "segments_${TASK}_final.txt"
exit 1
