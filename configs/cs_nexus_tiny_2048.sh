#!/bin/bash
# Flagship Chunk-Segmented Nilpotent Expansion Engine Configuration
PYTHONPATH=. torchrun --nproc_per_node=2 runs/main.py \
    --model cs_nexus --model_version tiny --seq_len 2048 --batch_size 8 --grad_accum 2 \
    --d_model 256 --num_heads 16 --num_layers 8 --chunk_size 32 --max_steps 30000