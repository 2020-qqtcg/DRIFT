#!/usr/bin/env bash
set -euo pipefail

# Edit these values as needed.
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-3B-Instruct"
OUTPUT_JSONL="./data/qwen2.5-3b-instruct/num1800_traj16_max512.jsonl"
DATASET_PATH="meta-math/MetaMathQA"
DATASET_SPLIT="train"
TYPE_PREFIX="MATH_"
MAX_PROMPTS="1800"
N_TRAJ=16
BATCH_SIZE=256
MAX_TOKENS=512
TEMPERATURE=1.0
TOP_P=1.0
SEED=42
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.85
MAX_TURNS=5
MAX_ACTIONS_PER_TRAJ=5
MAX_ACTIONS_PER_TURN=1
FORMAT_PENALTY=-0.1
INSTRUCTION_MAX_TOKENS=400
ACTION_SEP="||"
DISABLE_THINK=0
USE_CHAT_TEMPLATE=1
TRUST_REMOTE_CODE=0

mkdir -p "$(dirname "$OUTPUT_JSONL")"

python src/generate_weighted_data_vllm_metamathqa.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_jsonl "$OUTPUT_JSONL" \
  --dataset_path "$DATASET_PATH" \
  --dataset_split "$DATASET_SPLIT" \
  --type_prefix "$TYPE_PREFIX" \
  --n "$N_TRAJ" \
  --batch_size "$BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --seed "$SEED" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --max_turns "$MAX_TURNS" \
  --max_actions_per_traj "$MAX_ACTIONS_PER_TRAJ" \
  --max_actions_per_turn "$MAX_ACTIONS_PER_TURN" \
  --format_penalty "$FORMAT_PENALTY" \
  --instruction_max_tokens "$INSTRUCTION_MAX_TOKENS" \
  --action_sep "$ACTION_SEP" \
  $( [[ -n "$MAX_PROMPTS" ]] && echo --max_prompts "$MAX_PROMPTS" ) \
  $( [[ "$DISABLE_THINK" == "1" ]] && echo --disable_think ) \
  $( [[ "$USE_CHAT_TEMPLATE" == "1" ]] && echo --use_chat_template ) \
  $( [[ "$TRUST_REMOTE_CODE" == "1" ]] && echo --trust_remote_code )
