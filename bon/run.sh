#!/usr/bin/env bash

# source /home/sohyun0423/project/reward-model-bias/.venv-sd3/bin/activate
export CUDA_VISIBLE_DEVICES=${1:-0}

python -m bon.main --config bon/configs/default.yaml "${@:2}"
