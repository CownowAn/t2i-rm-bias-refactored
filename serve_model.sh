model=Qwen/Qwen3.5-9B
gpu=${1:-1,2,3,4}   # GPUs with enough free memory (avoid 0,5,6,7)

export CUDA_VISIBLE_DEVICES=$gpu
export HF_HOME=/nfs/data/sohyun/models
export TMPDIR=/nfs/data/sohyun/tmp
export TRITON_CACHE_DIR=/nfs/data/sohyun/triton_cache
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

# vllm 0.19.1 on NFS venv
#   - torch 2.10.0+cu128: CUDA 12.8 native (no libcudart.so.13 hack needed)
#   - SM86 (A6000) works: no SM86 CUBIN but CUDA driver handles compatibility
#   - Qwen3_5ForConditionalGeneration supported
#   - XFORMERS backend: bypasses FA2 SM80/SM90 limitation
#   - --enforce-eager: skip cudagraph (safer on non-standard SM)
source /nfs/data/sohyun/venvs/vllm-qwen35/bin/activate

VLLM_ATTENTION_BACKEND=XFORMERS vllm serve "$model" \
  --port 8000 \
  -dp 4 \
  --enforce-eager \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching

# vllm serve Qwen/Qwen3.5-9B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3