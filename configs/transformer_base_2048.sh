#!/bin/bash
PYTHONPATH=. torchrun --nproc_per_node=2 runs/main.py \
    --model transformer --model_version base --seq_len 2048 --batch_size 8 --grad_accum 2 \
    --d_model 768 --num_heads 16 --num_layers 16 --grad_checkpointing --max_steps 30000