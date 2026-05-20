#!/bin/bash
# Score baseline manifests with a reward model.
#
# Usage:
#   bash baselines/run_score.sh --gpu <id> [options]
#
# Examples:
#   bash baselines/run_score.sh --gpu 0 --topics 0 1 2
#   bash baselines/run_score.sh --gpu 0 --reward_model pickscore
#   bash baselines/run_score.sh --gpu 0 --manifest data/baselines/mjhq/topic_0/black-forest-labs-FLUX.1-dev/manifest.json

set -euo pipefail
export TMPDIR=/nfs/data/sohyun/tmp
mkdir -p "$TMPDIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPU="0"
TOPIC_IDS="0 1 2 3 4 5 6 7 8 9"
MANIFEST=""           # if set, score this single manifest directly (ignores topic_ids)
MODEL_ID="black-forest-labs/FLUX.1-dev"
OUTPUT_DIR="/nfs/data/sohyun/baselines/mjhq"
REWARD_MODEL="imagereward"
HF_CACHE_DIR="/nfs/data/sohyun/models"
BATCH_SIZE=32

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)          GPU="$2";           shift 2 ;;
        --topics)
            TOPIC_IDS=""
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TOPIC_IDS="$TOPIC_IDS $1"; shift
            done
            TOPIC_IDS="${TOPIC_IDS# }"
            ;;
        --manifest)     MANIFEST="$2";      shift 2 ;;
        --model_id)     MODEL_ID="$2";      shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --reward_model) REWARD_MODEL="$2";  shift 2 ;;
        --hf_cache_dir) HF_CACHE_DIR="$2";  shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_DIR_NAME=$(echo "$MODEL_ID" | tr '/' '-')
export CUDA_VISIBLE_DEVICES=$GPU

# ── Score ─────────────────────────────────────────────────────────────────────
if [[ -n "$MANIFEST" ]]; then
    # Direct manifest path provided
    echo "Scoring: $MANIFEST"
    python baselines/score_baselines.py \
        --manifest_path "$MANIFEST" \
        --reward_model "$REWARD_MODEL" \
        --device cuda:0 \
        --hf_cache_dir "$HF_CACHE_DIR" \
        --batch_size "$BATCH_SIZE"
else
    # Score all requested topics
    read -ra ALL_TOPICS <<< "$TOPIC_IDS"
    echo "=== Scoring (GPU $GPU, reward=$REWARD_MODEL) ==="
    echo "  Topics: ${ALL_TOPICS[*]}"
    echo ""
    for tid in "${ALL_TOPICS[@]}"; do
        MANIFEST_PATH="$OUTPUT_DIR/topic_${tid}/${MODEL_DIR_NAME}/manifest.json"
        if [[ ! -f "$MANIFEST_PATH" ]]; then
            echo "  WARNING: manifest not found for topic $tid: $MANIFEST_PATH"
            continue
        fi
        echo "  Topic $tid: $MANIFEST_PATH"
        python baselines/score_baselines.py \
            --manifest_path "$MANIFEST_PATH" \
            --reward_model "$REWARD_MODEL" \
            --device cuda:0 \
            --hf_cache_dir "$HF_CACHE_DIR" \
            --batch_size "$BATCH_SIZE"
    done
    echo "Done."
fi
