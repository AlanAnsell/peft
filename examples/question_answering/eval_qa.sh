#!/bin/bash
LANG=el

python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --peft_name_or_path models/xlmr_squad \
  --dataset_name xquad \
  --dataset_config_name xquad.${LANG} \
  --output_dir results/$LANG \
  --do_eval \
  --per_device_eval_batch_size 16 \
  --overwrite_output_dir
