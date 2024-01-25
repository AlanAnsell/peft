#!/bin/bash
MODEL_DIR=models/xlmr_squad
mkdir -p $MODEL_DIR

python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name squad \
  --output_dir $MODEL_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --overwrite_output_dir \
  --train_epochs 2 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --density 0.04 \
  --learning_rate 3e-5 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 2
