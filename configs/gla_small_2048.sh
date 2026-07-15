#!/bin/bash
PYTHONPATH=. torchrun --nproc_per_node=2 runs/main.py \
    --model gla --model_version small --seq_len 2048 --batch_size 8 --grad_accum 2 \
    --d_model 512 --num_heads 8 --num_layers 12 --grad_checkpointing --max_steps 30000