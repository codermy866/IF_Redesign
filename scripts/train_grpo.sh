#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
grpo_data="${1:-$root_dir/results/pilot_v0/counterfactual/grpo_train.jsonl}"
sft_adapter="${2:-$root_dir/results/pilot_v0/checkpoints/sft}"
if [[ ! -f "$grpo_data" ]]; then
  echo "Missing reviewed counterfactual GRPO dataset: $grpo_data" >&2
  exit 2
fi

export PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$root_dir/.venv/bin/swift" rlhf \
  --rlhf_type grpo \
  --model "$root_dir/models/pretrained/Qwen3-VL-2B-Instruct" \
  --adapters "$sft_adapter" \
  --dataset "$grpo_data" \
  --external_plugins "$root_dir/scripts/swift_reward_plugin.py" \
  --reward_funcs cervix_format cervix_cognition cervix_diagnosis \
  --reward_weights 1 1 2 \
  --train_type lora \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --num_generations 4 \
  --beta 0.04 \
  --learning_rate 1e-6 \
  --max_completion_length 384 \
  --max_length 2048 \
  --max_pixels 602112 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 50 \
  --logging_steps 1 \
  --save_steps 25 \
  --seed 20260902 \
  --output_dir "$root_dir/results/pilot_v0/checkpoints/grpo" \
  --add_version false \
  --report_to none
