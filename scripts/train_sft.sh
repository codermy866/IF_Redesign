#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$root_dir/.venv/bin/swift" sft \
  --model "$root_dir/models/pretrained/Qwen3-VL-2B-Instruct" \
  --dataset "$root_dir/results/pilot_v0/sft/train.jsonl" \
  --val_dataset "$root_dir/results/pilot_v0/sft/val.jsonl" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --max_length 2048 \
  --max_pixels 602112 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --logging_steps 1 \
  --dataset_num_proc 2 \
  --dataloader_num_workers 2 \
  --seed 20260902 \
  --data_seed 20260902 \
  --output_dir "$root_dir/results/pilot_v0/checkpoints/sft" \
  --add_version false \
  --report_to none
