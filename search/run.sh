# source /home/sohyun0423/project/reward-model-bias/.venv-sd3/bin/activate
export CUDA_VISIBLE_DEVICES=$1

# Build flux_devices=[cuda:0,cuda:1,...] from CUDA_VISIBLE_DEVICES
N_GPUS=$(echo "${1:-0}" | tr ',' '\n' | wc -l | tr -d ' ')
FLUX_DEVICES=$(python3 -c "n=$N_GPUS; print('[' + ','.join(f'cuda:{i}' for i in range(n)) + ']')")

# python -m search.main --config search/configs/debug.yaml "models.editor.flux_devices=$FLUX_DEVICES"
# python -m search.main --config search/configs/default.yaml "models.editor.flux_devices=$FLUX_DEVICES"
python -m search.main --config search/configs/residual.yaml "models.editor.flux_devices=$FLUX_DEVICES"