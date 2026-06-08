#!/bin/bash
# Editor capability survey for counterfactual debiasing PoC.
#
# Usage:
#   bash debias/counterfactual/poc/run_poc.sh \
#       --gpus 0 \
#       --per_prompt_W_path outputs/search/<RUN>/per_prompt_W_step<N>_topic<T>.json \
#       --topic_id 0
#
# Smoke test (3-5 min):
#   bash debias/counterfactual/poc/run_poc.sh \
#       --gpus 0 \
#       --per_prompt_W_path outputs/search/20260601-154143/per_prompt_W_step12_topic0.json \
#       --topic_id 0 \
#       --run_id smoke_test \
#       --tau 0.05 --top_n_per_prompt 2 --n_prompts_per_attr 2 --n_images_per_prompt 1

set -euo pipefail
export TMPDIR=/nfs/data/sohyun/tmp
mkdir -p "$TMPDIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS="0"
PYTHON="python"
RUN_ID=""

PER_PROMPT_W_PATH=""
BA_EXPAND_PATH=""
TOPIC_ID="0"
DETECTION_CACHE_PATH="outputs/detection_cache/mjhq/black-forest-labs-FLUX.1-dev.json"
DETECTOR_KEY="Qwen/Qwen3.5-9B::auto"
BASELINE_MANIFEST=""
BASELINE_ROOT=""
PROMPTS_DIR="clustering/output/mjhq"

TAU="0.0"
TOP_N_PER_PROMPT="5"
N_PROMPTS_PER_ATTR="100"
N_IMAGES_PER_PROMPT="10"
HUMANNESS_RECHECK="false"
HUMANNESS_MODEL="openai/gpt-5"

FLUX_MODEL="black-forest-labs/FLUX.1-Kontext-dev"
EDITOR_DEVICES=""              # auto-derived from --gpus if empty
EDITOR_MAX_PARALLEL=""         # default: = number of editor devices
GUIDANCE_SCALE="2.5"
HF_CACHE_DIR="/nfs/data/sohyun/models"

DETECTOR_MODEL="Qwen/Qwen3.5-9B"
DETECTOR_VLLM_BASE_URL=""
DETECTOR_IMAGE_DETAIL="auto"
DETECTOR_MAX_PARALLEL="32"

SIDE_EFFECT_CHECK="false"
MAKE_THUMBNAILS="true"
CHECK_REWARD="false"
REWARD_MODEL="imagereward"        # "imagereward" | "pickscore" | "hpsv3"
REWARD_DEVICE=""                  # default: first editor device
REWARD_HF_CACHE_DIR="/nfs/data/sohyun/models"
SOURCE_CONSISTENCY_N="0"          # 0 = skip; e.g. 3 = require g=1 in all 3 detector re-queries
INSTRUCTION_MODE="correct"        # "correct" | "remove"; correct avoids the "object erased" failure mode
LIMIT_N_ATTRS="0"                 # 0 = no limit; >0 = only first N attrs of per_prompt_W
KEEP_ALIVE="false"
SEED="42"
CF_ROOT="/nfs/data/sohyun/projects/t2i-rm-bias/counterfactuals"
REPORT_ROOT="outputs/counterfactual_poc"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)                   GPUS="$2"; shift 2 ;;
        --python)                 PYTHON="$2"; shift 2 ;;
        --run_id)                 RUN_ID="$2"; shift 2 ;;
        --per_prompt_W_path)      PER_PROMPT_W_PATH="$2"; shift 2 ;;
        --ba_expand_path)         BA_EXPAND_PATH="$2"; shift 2 ;;
        --topic_id)               TOPIC_ID="$2"; shift 2 ;;
        --detection_cache_path)   DETECTION_CACHE_PATH="$2"; shift 2 ;;
        --detector_key)           DETECTOR_KEY="$2"; shift 2 ;;
        --baseline_manifest)      BASELINE_MANIFEST="$2"; shift 2 ;;
        --baseline_root)          BASELINE_ROOT="$2"; shift 2 ;;
        --prompts_dir)            PROMPTS_DIR="$2"; shift 2 ;;
        --tau)                    TAU="$2"; shift 2 ;;
        --top_n_per_prompt)       TOP_N_PER_PROMPT="$2"; shift 2 ;;
        --n_prompts_per_attr)     N_PROMPTS_PER_ATTR="$2"; shift 2 ;;
        --n_images_per_prompt)    N_IMAGES_PER_PROMPT="$2"; shift 2 ;;
        --humanness_recheck)      HUMANNESS_RECHECK="true"; shift 1 ;;
        --humanness_model)        HUMANNESS_MODEL="$2"; shift 2 ;;
        --flux_model)             FLUX_MODEL="$2"; shift 2 ;;
        --editor_devices)         EDITOR_DEVICES="$2"; shift 2 ;;
        --editor_max_parallel)    EDITOR_MAX_PARALLEL="$2"; shift 2 ;;
        --guidance_scale)         GUIDANCE_SCALE="$2"; shift 2 ;;
        --hf_cache_dir)           HF_CACHE_DIR="$2"; shift 2 ;;
        --detector_model)         DETECTOR_MODEL="$2"; shift 2 ;;
        --detector_vllm_base_url) DETECTOR_VLLM_BASE_URL="$2"; shift 2 ;;
        --detector_image_detail)  DETECTOR_IMAGE_DETAIL="$2"; shift 2 ;;
        --detector_max_parallel)  DETECTOR_MAX_PARALLEL="$2"; shift 2 ;;
        --side_effect_check)      SIDE_EFFECT_CHECK="true"; shift 1 ;;
        --make_thumbnails)        MAKE_THUMBNAILS="true"; shift 1 ;;
        --no_thumbnails)          MAKE_THUMBNAILS="false"; shift 1 ;;
        --check_reward)           CHECK_REWARD="true"; shift 1 ;;
        --reward_model)           REWARD_MODEL="$2"; shift 2 ;;
        --reward_device)          REWARD_DEVICE="$2"; shift 2 ;;
        --reward_hf_cache_dir)    REWARD_HF_CACHE_DIR="$2"; shift 2 ;;
        --source_consistency_n)   SOURCE_CONSISTENCY_N="$2"; shift 2 ;;
        --instruction_mode)       INSTRUCTION_MODE="$2"; shift 2 ;;
        --limit_n_attrs)          LIMIT_N_ATTRS="$2"; shift 2 ;;
        --keep_alive)             KEEP_ALIVE="true"; shift 1 ;;
        --seed)                   SEED="$2"; shift 2 ;;
        --cf_root)                CF_ROOT="$2"; shift 2 ;;
        --report_root)            REPORT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$PER_PROMPT_W_PATH" ]]; then
    echo "ERROR: --per_prompt_W_path is required"; exit 1
fi
if [[ -z "$BASELINE_MANIFEST" ]]; then
    echo "ERROR: --baseline_manifest is required"; exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
# Inside the process, CUDA renumbers visible devices to start at 0.
# Auto-derive editor device list from the GPU count unless caller overrode it.
if [[ -z "$EDITOR_DEVICES" ]]; then
    N_GPUS=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
    EDITOR_DEVICES=$(python3 -c "n=$N_GPUS; print(','.join(f'cuda:{i}' for i in range(n)))")
fi

ARGS=(
    --per_prompt_W_path "$PER_PROMPT_W_PATH"
    --topic_id "$TOPIC_ID"
    --detection_cache_path "$DETECTION_CACHE_PATH"
    --detector_key "$DETECTOR_KEY"
    --baseline_manifest "$BASELINE_MANIFEST"
    --baseline_root "$BASELINE_ROOT"
    --tau "$TAU"
    --top_n_per_prompt "$TOP_N_PER_PROMPT"
    --n_prompts_per_attr "$N_PROMPTS_PER_ATTR"
    --n_images_per_prompt "$N_IMAGES_PER_PROMPT"
    --humanness_model "$HUMANNESS_MODEL"
    --flux_model "$FLUX_MODEL"
    --editor_devices "$EDITOR_DEVICES"
    --guidance_scale "$GUIDANCE_SCALE"
    --hf_cache_dir "$HF_CACHE_DIR"
    --detector_model "$DETECTOR_MODEL"
    --detector_image_detail "$DETECTOR_IMAGE_DETAIL"
    --detector_max_parallel "$DETECTOR_MAX_PARALLEL"
    --source_consistency_n "$SOURCE_CONSISTENCY_N"
    --instruction_mode "$INSTRUCTION_MODE"
    --limit_n_attrs "$LIMIT_N_ATTRS"
    --seed "$SEED"
    --cf_root "$CF_ROOT"
    --report_root "$REPORT_ROOT"
)
if [[ -n "$EDITOR_MAX_PARALLEL" ]]; then ARGS+=(--editor_max_parallel "$EDITOR_MAX_PARALLEL"); fi
if [[ -n "$RUN_ID" ]]; then              ARGS+=(--run_id "$RUN_ID"); fi
if [[ -n "$BA_EXPAND_PATH" ]]; then      ARGS+=(--ba_expand_path "$BA_EXPAND_PATH"); fi
if [[ "$HUMANNESS_RECHECK" == "true" ]]; then ARGS+=(--humanness_recheck); fi
if [[ "$SIDE_EFFECT_CHECK"  == "true" ]]; then ARGS+=(--side_effect_check); fi
if [[ "$MAKE_THUMBNAILS"    == "true" ]]; then ARGS+=(--make_thumbnails); fi
if [[ "$KEEP_ALIVE"         == "true" ]]; then ARGS+=(--keep_alive); fi
if [[ "$CHECK_REWARD"       == "true" ]]; then
    ARGS+=(--check_reward --reward_model "$REWARD_MODEL" \
           --reward_hf_cache_dir "$REWARD_HF_CACHE_DIR")
    if [[ -n "$REWARD_DEVICE" ]]; then ARGS+=(--reward_device "$REWARD_DEVICE"); fi
fi
if [[ -n "$DETECTOR_VLLM_BASE_URL" ]]; then ARGS+=(--detector_vllm_base_url "$DETECTOR_VLLM_BASE_URL"); fi

echo "=== Counterfactual PoC ==="
echo "  GPUs           : $GPUS  →  editor_devices=$EDITOR_DEVICES"
echo "  PerPromptW     : $PER_PROMPT_W_PATH"
echo "  TopicID        : $TOPIC_ID"
echo "  Tau            : $TAU"
echo "  Top-N / prompt : $TOP_N_PER_PROMPT"
echo "  Sub-sample     : $N_PROMPTS_PER_ATTR prompts × $N_IMAGES_PER_PROMPT imgs per attr"
echo "  RecheckHumman  : $HUMANNESS_RECHECK"
echo "  SideEffectChk  : $SIDE_EFFECT_CHECK"
echo "  EditorDevices  : $EDITOR_DEVICES"
echo "  CheckReward    : $CHECK_REWARD${CHECK_REWARD:+ ($REWARD_MODEL${REWARD_DEVICE:+ on $REWARD_DEVICE})}"
echo "  SourceConsist. : $SOURCE_CONSISTENCY_N${SOURCE_CONSISTENCY_N:+ rounds (0 = off)}"
echo "  InstructionMode: $INSTRUCTION_MODE"
echo "  LimitNAttrs    : $LIMIT_N_ATTRS${LIMIT_N_ATTRS:+ (0 = all)}"
echo "  KeepAlive      : $KEEP_ALIVE"
echo "  CF root        : $CF_ROOT"
echo "  Report root    : $REPORT_ROOT"
echo ""

exec $PYTHON -u -m debias.counterfactual.poc.run_poc "${ARGS[@]}"
