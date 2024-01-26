#!/bin/bash
MODEL_DIR=models/xlmr_squad_sm3
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
  --num_train_epochs 2 \
  --logging_steps 5 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --selection_algorithm sm3 \
  --density 0.04 \
  --learning_rate 3e-3 \
  --warmup_ratio 0.1 \
  --weight_decay 10 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 2 > $MODEL_DIR/log.txt 2>&1
