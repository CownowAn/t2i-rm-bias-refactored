#!/usr/bin/env bash

# source /home/sohyun0423/project/reward-model-bias/.venv-sd3/bin/activate
export CUDA_VISIBLE_DEVICES=${1:-0}

# python -m bon.main --config bon/configs/default.yaml "${@:2}"
python -m bon.main --config bon/configs/default_mjhq.yaml "${@:2}"


# bash bon/run.sh 1 "attributes.search_results_path=/home/sohyun0423/project/t2i-rm-bias/outputs/search/20260518-235212/results.json" "models.reward_model.name=hpsv3"
# bash bon/run.sh 2 "attributes.search_results_path=/home/sohyun0423/project/t2i-rm-bias/outputs/search/20260519-152519/results.json" "models.reward_model.name=imagereward" "data.baseline_manifest=/nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_2/black-forest-labs-FLUX.1-dev/manifest.json" "data.topic_ids=[2]"