#!/bin/bash
PYTHONPATH=. torchrun --nproc_per_node=2 runs/main.py \
    --model gdn2 --model_version base --seq_len 2048 --batch_size 8 --grad_accum 2 \
    --d_model 768 --num_heads 10 --num_layers 10 --grad_checkpointing --max_steps 30000