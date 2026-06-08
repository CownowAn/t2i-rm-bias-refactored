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
# TOPIC_IDS="0 1 2 3 4 5 6 7 8 9"
TOPIC_IDS="0 3 5 6 8 9 1 2 4 7"
MODEL_ID="black-forest-labs/FLUX.1-dev"
MODEL_ID="stabilityai/stable-diffusion-3.5-medium"
# CLUSTER_DIR="clustering/output/mjhq"
CLUSTER_DIR="clustering/output/mjhq_10tok/100prompt"
OUTPUT_DIR="/nfs/data/sohyun/projects/t2i-rm-bias/data/baselines/mjhq_10tok/100prompt"
HF_CACHE_DIR="/nfs/data/sohyun/models"
IMAGES_PER_PROMPT=128
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
PYTHON="python"   # override with --python /path/to/venv/bin/python if needed
KEEP_ALIVE=0      # set to 1 with --keep_alive: hold pipelines on GPU after merge

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
        --keep_alive)        KEEP_ALIVE=1;            shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required (e.g. --gpus 0,1,2,3)"
    exit 1
fi

# SD 2.1 (768 v-pred) — auto-bump resolution; -base variants stay at 512.
if [[ "$MODEL_ID" == *stable-diffusion-2* && "$MODEL_ID" != *-base ]]; then
    IMAGE_WIDTH=768
    IMAGE_HEIGHT=768
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
echo "  Res:     ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo "  Output:  $OUTPUT_DIR"
echo "  Mode:    one python per GPU; each handles all topics; FLUX loaded once"
[[ $KEEP_ALIVE -eq 1 ]] && echo "  Keep:    pipelines stay on GPU after merge (Ctrl+C to release)"
echo ""

# Done-flag dir for keep-alive mode: pythons touch a file here when their
# topic loop is finished but BEFORE idling, so the shell can run the manifest
# merge without having to wait on the (still-running) python process.
DONE_DIR="$LOG_DIR/.done_flags"
KEEP_ALIVE_ARGS=()
if [[ $KEEP_ALIVE -eq 1 ]]; then
    rm -rf "$DONE_DIR"
    mkdir -p "$DONE_DIR"
    KEEP_ALIVE_ARGS=(--keep_alive --done_flag_dir "$DONE_DIR")
fi

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
        "${KEEP_ALIVE_ARGS[@]}" \
        2>&1 | sed -u "s/^/[GPU $gpu | shard $rank] /" | tee "$log" &

    PIDS+=($!)
done

FAILED=0
if [[ $KEEP_ALIVE -eq 1 ]]; then
    # Poll for done flags instead of `wait`ing on PIDs (which would block
    # forever because pythons idle on GPU after generation).
    echo "Waiting for all $N_GPUS shards to write done flags in $DONE_DIR ..."
    while true; do
        n_done=0
        for rank in "${!GPU_LIST[@]}"; do
            [[ -e "$DONE_DIR/shard_${rank}.done" ]] && n_done=$((n_done + 1))
        done
        if [[ $n_done -eq $N_GPUS ]]; then break; fi

        # Detect a crashed shard (process gone without writing its flag).
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"; gpu="${GPU_LIST[$i]}"
            if ! kill -0 "$pid" 2>/dev/null \
               && [[ ! -e "$DONE_DIR/shard_${i}.done" ]]; then
                echo "  GPU $gpu (shard $i): FAILED (see $LOG_DIR/gpu${gpu}.log)"
                FAILED=1
            fi
        done
        [[ $FAILED -eq 1 ]] && break
        sleep 5
    done
    if [[ $FAILED -ne 1 ]]; then
        echo "All shards reported done. Proceeding to merge while pipelines stay loaded."
    fi
else
    # Wait for all shards in the usual (non-keep-alive) path.
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"; gpu="${GPU_LIST[$i]}"
        if wait "$pid"; then
            echo "  GPU $gpu: done"
        else
            echo "  GPU $gpu: FAILED (see $LOG_DIR/gpu${gpu}.log)"
            FAILED=1
        fi
    done
fi

if [[ $FAILED -eq 1 ]]; then
    [[ $KEEP_ALIVE -eq 1 ]] && { echo "Killing surviving shards ..."; kill "${PIDS[@]}" 2>/dev/null; }
    echo "Aborting due to failure."
    exit 1
fi

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

if [[ $KEEP_ALIVE -eq 1 ]]; then
    echo ""
    echo "── Keep-alive mode ──────────────────────────────────────────────────────"
    echo "Pipelines remain loaded on GPUs: ${GPU_LIST[*]}"
    echo "Press Ctrl+C (or kill this shell) to release."
    trap 'echo ""; echo "Releasing GPUs ..."; kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; exit 0' INT TERM
    wait
fi
