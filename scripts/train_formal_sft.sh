#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FOLD VARIANT SEED" >&2
  exit 2
fi

fold="$1"
variant="$2"
seed="$3"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}"

data_dir="$root_dir/results/formal_v1/folds/$fold/variants/$variant"
runs_root="${FORMAL_RUNS_ROOT:-$root_dir/results/formal_v1/runs}"
output_dir="$runs_root/$variant/$fold/seed_$seed/checkpoints"
if [[ ! -f "$data_dir/train_sft.jsonl" || ! -f "$data_dir/val_sft.jsonl" ]]; then
  echo "missing formal SFT data under $data_dir" >&2
  exit 3
fi

# Interrupted jobs retain their latest epoch checkpoint.  Automatic resume is
# opt-in because a partially retained checkpoint can be missing adapter weights
# after an interrupted save; a fresh deterministic rerun is safer by default.
resume_args=()
if [[ "${FORMAL_RESUME_FROM_LATEST:-false}" == "true" && -d "$output_dir" ]]; then
  latest_checkpoint="$(find "$output_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -t- -k2,2n | tail -n 1)"
  if [[ -n "$latest_checkpoint" && -f "$output_dir/$latest_checkpoint/adapter_model.safetensors" ]]; then
    resume_path="$output_dir/$latest_checkpoint"
    echo "resuming formal SFT from $resume_path"
    resume_args=(--resume_from_checkpoint "$resume_path")
  elif [[ -n "$latest_checkpoint" ]]; then
    echo "latest checkpoint lacks adapter weights; refusing resume" >&2
  fi
fi

"$root_dir/.venv/bin/swift" sft \
  --model "$root_dir/models/pretrained/Qwen3-VL-2B-Instruct" \
  --dataset "$data_dir/train_sft.jsonl" \
  --val_dataset "$data_dir/val_sft.jsonl" \
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
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing true \
  --max_length 2048 \
  --max_pixels 602112 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_steps 10 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --seed "$seed" \
  --data_seed "$seed" \
  --output_dir "$output_dir" \
  "${resume_args[@]}" \
  --add_version false \
  --report_to none

if [[ "${FORMAL_SKIP_COMPLETION_MARKER:-false}" != "true" ]]; then
  PYTHONPATH="$root_dir/src:$root_dir/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$root_dir/.venv/bin/python" -c \
    "from run_formal_training_queue import mark_complete; mark_complete('$variant', '$fold', int('$seed'))"
fi
