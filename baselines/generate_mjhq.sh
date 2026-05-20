#!/bin/bash
# Generate MJHQ-30K baseline images using a T2I model.
#
# One python process per GPU; each process handles ALL requested topics in
# sequence, with the FLUX pipe loaded once and reused across topics. This way
# a shard whose assigned prompts are already complete for some topic still
# keeps the model on its GPU for later topics — no per-topic reload.
#
# Usage:
#   bash baselines/generate_mjhq.sh --gpus <gpu_ids> [--topics <topic_ids>]
#
# Examples:
#   bash baselines/generate_mjhq.sh --gpus 0              # single GPU, all topics
#   bash baselines/generate_mjhq.sh --gpus 0,1,2,3        # 4 GPUs share each topic's prompts
#   bash baselines/generate_mjhq.sh --gpus 0,1 --topics 0 # GPU 0 & 1 split topic 0's prompts

set -euo pipefail
export TMPDIR=/nfs/data/sohyun/tmp
mkdir -p "$TMPDIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=""
TOPIC_IDS="0 1 2 3 4 5 6 7 8 9"
MODEL_ID="black-forest-labs/FLUX.1-dev"
CLUSTER_DIR="clustering/output/mjhq"
OUTPUT_DIR="/nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq"
HF_CACHE_DIR="/nfs/data/sohyun/models"
IMAGES_PER_PROMPT=128
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
PYTHON="python"   # override with --python /path/to/venv/bin/python if needed

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)    GPUS="$2"; shift 2 ;;
        --topics)
            TOPIC_IDS=""
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TOPIC_IDS="$TOPIC_IDS $1"; shift
            done
            TOPIC_IDS="${TOPIC_IDS# }"
            ;;
        --model_id)          MODEL_ID="$2";          shift 2 ;;
        --cluster_dir)       CLUSTER_DIR="$2";        shift 2 ;;
        --output_dir)        OUTPUT_DIR="$2";         shift 2 ;;
        --images_per_prompt) IMAGES_PER_PROMPT="$2";  shift 2 ;;
        --python)            PYTHON="$2";             shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required (e.g. --gpus 0,1,2,3)"
    exit 1
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}
read -ra ALL_TOPICS <<< "$TOPIC_IDS"
MODEL_DIR_NAME=$(echo "$MODEL_ID" | tr '/' '-')
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== MJHQ Baseline Generation ==="
echo "  GPUs:    ${GPU_LIST[*]} (${N_GPUS} total)"
echo "  Topics:  ${ALL_TOPICS[*]}"
echo "  Model:   $MODEL_ID"
echo "  Output:  $OUTPUT_DIR"
echo "  Mode:    one python per GPU; each handles all topics; FLUX loaded once"
echo ""

# ── Launch one shard per GPU; each processes ALL topics in sequence ──────────
PIDS=()
for rank in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$rank]}"
    log="$LOG_DIR/gpu${gpu}.log"
    echo "  GPU $gpu (shard $rank/$N_GPUS) → $log"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u baselines/generate_images.py \
        --cluster_dir "$CLUSTER_DIR" \
        --topic_ids ${ALL_TOPICS[@]} \
        --output_dir "$OUTPUT_DIR" \
        --model_id "$MODEL_ID" \
        --images_per_prompt $IMAGES_PER_PROMPT \
        --image_width $IMAGE_WIDTH \
        --image_height $IMAGE_HEIGHT \
        --hf_cache_dir "$HF_CACHE_DIR" \
        --num_shards $N_GPUS \
        --shard_rank $rank \
        2>&1 | sed -u "s/^/[GPU $gpu | shard $rank] /" | tee "$log" &

    PIDS+=($!)
done

# Wait for all shards
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"; gpu="${GPU_LIST[$i]}"
    if wait "$pid"; then
        echo "  GPU $gpu: done"
    else
        echo "  GPU $gpu: FAILED (see $LOG_DIR/gpu${gpu}.log)"
        FAILED=1
    fi
done

[[ $FAILED -eq 1 ]] && { echo "Aborting due to failure."; exit 1; }

# ── Merge shard manifests per topic ───────────────────────────────────────────
for topic_id in "${ALL_TOPICS[@]}"; do
    OUT_DIR_TOPIC="$OUTPUT_DIR/topic_${topic_id}/${MODEL_DIR_NAME}"
    echo "  Topic $topic_id: merging ${N_GPUS} shard manifests ..."
    $PYTHON - <<PYEOF
import json, sys
from pathlib import Path
out_dir = Path("$OUT_DIR_TOPIC")
num_shards = $N_GPUS

merged = {}
metadata = {}
for rank in range(num_shards):
    p = out_dir / f"manifest_shard_{rank}.json"
    if not p.exists():
        print(f"  WARNING: missing shard {rank}: {p}", file=sys.stderr)
        continue
    d = json.loads(p.read_text())
    metadata = d.get("metadata", metadata)
    merged.update(d.get("baselines", {}))

manifest_path = out_dir / "manifest.json"
manifest_path.write_text(json.dumps({"metadata": metadata, "baselines": merged}, indent=2, ensure_ascii=False))
print(f"  Merged {len(merged)} prompts → {manifest_path}")
PYEOF
done

echo ""
echo "All topics done. Next: run scoring."
echo ""
echo "  bash baselines/run_score.sh --gpu ${GPU_LIST[0]} --output_dir $OUTPUT_DIR"
echo ""
echo "Then update bon_amplified.yaml:"
echo "  data.baseline_manifest: $OUTPUT_DIR/topic_0/${MODEL_DIR_NAME}/manifest.json"
echo "  data.baseline_root: $(pwd)"
echo "  data.topic_ids: [0]"
