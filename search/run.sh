#!/bin/bash
# Run the BoN-amplified search.
#
# Usage:
#   bash search/run.sh <gpu_id> [key=value overrides...]
#
# Example:
#   bash search/run.sh 7 models.reward_model.name=pickscore data.topic_ids=[0]
#
# The active config is search/configs/bon_amplified_partial.yaml; see
# search/configs/bon_amplified_debug.yaml for a cheap N=2 smoke-test config.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$1
export TMPDIR=/nfs/data/sohyun/tmp
mkdir -p "$TMPDIR"

python -m search.main --config search/configs/bon_amplified_partial.yaml "${@:2}"
