#!/bin/bash
PYTHONPATH=. torchrun --nproc_per_node=2 runs/main.py \
    --model transformer --model_version tiny --seq_len 256 --batch_size 64 --grad_accum 2 \
    --d_model 256 --num_heads 16 --num_layers 8 --max_steps 30000