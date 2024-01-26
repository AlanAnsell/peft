#!/bin/bash
MODEL_DIR=models/xlmr_squad_rigl_4bit
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
  --load_in_4bit yes \
  --selection_algorithm rigl \
  --density 0.04 \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 2 #> $MODEL_DIR/log.txt 2>&1
