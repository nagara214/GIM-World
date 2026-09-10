#!/bin/bash
# Quick start: roll out the bundled example clip with GIM-World.
# Edit the variables below, then `bash run.sh`.

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# Paths printed by `python download_models.py`
WAN_DIR=${WAN_DIR:-checkpoints/Wan2.1-T2V-1.3B}
GIM_CKPT=${GIM_CKPT:-checkpoints/GIM-World/first_person}   # or .../third_person

INPUT=${INPUT:-assets/examples/demo_scene}   # folder with video.mp4 + action.json
OUTPUT=${OUTPUT:-outputs/demo}
MAX_SECONDS=${MAX_SECONDS:-20}               # rollout length after mark_time
NUM_STEPS=${NUM_STEPS:-50}
SEED=${SEED:-42}

python generate.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --gim_ckpt "$GIM_CKPT" \
    --wan_dir "$WAN_DIR" \
    --max_seconds "$MAX_SECONDS" \
    --num_steps "$NUM_STEPS" \
    --seed "$SEED" \
    --include_prefix \
    "$@"
