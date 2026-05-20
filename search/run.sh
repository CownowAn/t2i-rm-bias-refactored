# source /home/sohyun0423/project/reward-model-bias/.venv-sd3/bin/activate
export CUDA_VISIBLE_DEVICES=$1

export TMPDIR=/nfs/data/sohyun/tmp
mkdir -p "$TMPDIR"

# Build flux_devices=[cuda:0,cuda:1,...] from CUDA_VISIBLE_DEVICES
# N_GPUS=$(echo "${1:-0}" | tr ',' '\n' | wc -l | tr -d ' ')
# FLUX_DEVICES=$(python3 -c "n=$N_GPUS; print('[' + ','.join(f'cuda:{i}' for i in range(n)) + ']')")

# python -m search.main --config search/configs/debug.yaml "models.editor.flux_devices=$FLUX_DEVICES"
# python -m search.main --config search/configs/default.yaml "models.editor.flux_devices=$FLUX_DEVICES"
# python -m search.main --config search/configs/residual.yaml "models.editor.flux_devices=$FLUX_DEVICES"

# ── Baseline-pairs mode (no FLUX needed) ──
# python -m search.main --config search/configs/baseline_pairs_debug.yaml
# python -m search.main --config search/configs/baseline_pairs.yaml "${@:2}"

# ── BoN-Amplified mode (image-level U^{N-1} residuals, no FLUX needed) ──
# python -m search.main --config search/configs/bon_amplified.yaml "${@:2}"
# python -m search.main --config search/configs/bon_amplified.yaml
python -m search.main --config search/configs/bon_amplified_partial.yaml "${@:2}"


# # ImageReward (default)
# bash search/run.sh 7 "models.reward_model.name=imagereward" "data.topic_ids=[2]" "data.baseline_manifest=/nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq/topic_2/black-forest-labs-FLUX.1-dev/manifest.json"

# # PickScore
# bash search/run.sh 6 "models.reward_model.name=pickscore"

# # HPSv3
# bash search/run.sh 7 "models.reward_model.name=hpsv3"