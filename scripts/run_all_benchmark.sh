#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Multi-benchmark Math/QA Evaluation Script
# =============================================================================
# export CUDA_VISIBLE_DEVICES=1

# Model configuration
MODEL_NAME="drift" # Decide the filename of outputs
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-3B-Instruct"
LORA_PATH=""  # Leave empty if not using LoRA

# Benchmarks to run
DATASETS=(
  "math500"
  "theoremqa"
  "gpqa"
  "mmlu-redux"
  "hendrycks_math"
  "mmlu_pro"
)

# Output root (per-dataset subdir will be created)
OUTPUT_ROOT="results/Qwen2.5-3B-Instruct"

# Dataset-related (kept for compatibility; only used by some datasets in your code)
DATASET_PATH="meta-math/MetaMathQA"   # Only used when DATASET=metamathqa (per your comment)
DATASET_SPLIT="test"
TYPE_PREFIX="MATH_"                  # Only used when DATASET=metamathqa
DATASET_CONFIG="default"
MAX_SAMPLES=""                       # Leave empty to use all samples

# Generation parameters
MAX_TURNS=5
MAX_ACTIONS_PER_TRAJ=5
MAX_ACTIONS_PER_TURN=1
FORMAT_PENALTY=-0.1
INSTRUCTION_MAX_TOKENS=400
ACTION_SEP="||"
DISABLE_THINK=0
PROMPT_MODE="simple"   # "full" or "simple"
BATCH_SIZE=256
MAX_TOKENS=1024
TEMPERATURE=0
TOP_P=0.1
SEED=42

# Hardware configuration
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
MAX_NUM_SEQS=256  # Reduce if OOM with LoRA

# LoRA configuration
MAX_LORA_RANK=8
LORA_EXTRA_VOCAB_SIZE=0

# Other options
USE_CHAT_TEMPLATE=1
TRUST_REMOTE_CODE=0

# =============================================================================
# Do not edit below this line
# =============================================================================

run_one() {
  local dataset="$1"
  local output_json="${OUTPUT_ROOT}/${dataset}/${MODEL_NAME}.json"

  mkdir -p "$(dirname "$output_json")"

  echo "============================================================"
  echo "Running dataset: ${dataset}"
  echo "Output: ${output_json}"
  echo "============================================================"

  python src/eval_math_multiturn_vllm.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_json "$output_json" \
    --dataset "$dataset" \
    --dataset_path "$DATASET_PATH" \
    --dataset_split "$DATASET_SPLIT" \
    --type_prefix "$TYPE_PREFIX" \
    $( [[ -n "$DATASET_CONFIG" ]] && echo --dataset_config "$DATASET_CONFIG" ) \
    --max_turns "$MAX_TURNS" \
    --max_actions_per_traj "$MAX_ACTIONS_PER_TRAJ" \
    --max_actions_per_turn "$MAX_ACTIONS_PER_TURN" \
    --format_penalty "$FORMAT_PENALTY" \
    --instruction_max_tokens "$INSTRUCTION_MAX_TOKENS" \
    --action_sep "$ACTION_SEP" \
    --prompt_mode "$PROMPT_MODE" \
    --fix_mistral_regex \
    --batch_size "$BATCH_SIZE" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --seed "$SEED" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --max_lora_rank "$MAX_LORA_RANK" \
    --lora_extra_vocab_size "$LORA_EXTRA_VOCAB_SIZE" \
    $( [[ -n "$LORA_PATH" ]] && echo --lora_path "$LORA_PATH" ) \
    $( [[ -n "$MAX_SAMPLES" ]] && echo --max_samples "$MAX_SAMPLES" ) \
    $( [[ "$DISABLE_THINK" == "1" ]] && echo --disable_think ) \
    $( [[ "$USE_CHAT_TEMPLATE" == "1" ]] && echo --use_chat_template ) \
    $( [[ "$TRUST_REMOTE_CODE" == "1" ]] && echo --trust_remote_code )
}

for ds in "${DATASETS[@]}"; do
  run_one "$ds"
done

echo "All datasets done."
