import sys
import os
# Dynamically inject the parent directory (project root) into Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from datetime import timedelta
import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import AutoTokenizer
from models import make_model
from trainers.causal_trainer import CausalTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="BareTorch Foundation Model Execution Subsystem")
    
    # Core Experiment Setup
    parser.add_argument("--model", type=str, required=True, 
                        choices=["transformer", "gla", "gdn2", "mamba3", "cs_lrad", "ccrs", "ckts", "cbkc", "cofe"])
    parser.add_argument("--model_version", type=str, default="tiny", choices=["tiny", "small", "base"],
                        help="Model sizing tier variant scale allocation tracking category")
    parser.add_argument("--seq_len", type=int, default=2048, help="Context sequence length tracking window")
    
    # Unified Architecture Hyperparameters
    parser.add_argument("--d_model", type=int, default=256, help="Hidden embedding dimension capacity")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of tracking attention/mixing heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Total number of cascaded decoder layer blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout regularization probability factor")
    parser.add_argument("--grad_checkpointing", action="store_true", help="Enable memory-efficient gradient checkpointing")
    
    # Sub-Quadratic Gated State Engine Specifics
    parser.add_argument("--chunk_size", type=int, default=32, help="Sequence block chunk partitioning size for local GEMMs")
    parser.add_argument("--rank", type=int, default=8, help="Uniform low-rank bottleneck factor (r) for active delta engines")
    
    # Optimization & Trainer Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Local batch size per active GPU accelerator instance")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation evaluation steps mapping")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak target learning rate ceiling value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay regularizer constraint")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Linear scheduler optimization warmup count step limits")
    parser.add_argument("--max_steps", type=int, default=30000, help="Hard stop training step execution limit allocation")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize distributed process parameters
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)
    
    # Configure path strings automatically based on dynamic version run settings
    run_name = f"{args.model}_{args.model_version}_{args.seq_len}"
    
    if global_rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs(f"results/{run_name}", exist_ok=True)
    dist.barrier()
    
    # Calculate the absolute project root dynamically
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dataset_target_path = os.path.join(project_root, f"processed_wiki_dataset_{args.seq_len}")
    if not os.path.exists(dataset_target_path):
        fallback_path = os.path.join(project_root, "processed_wiki_dataset")
        if os.path.exists(fallback_path):
            dataset_target_path = fallback_path
        else:
            raise FileNotFoundError(
                f"BareTorch Data Error: Could not locate dataset folder at '{dataset_target_path}' "
                f"or fallback location '{fallback_path}'. Please run tokenize_wiki.py first."
            )
            
    processed_splits = load_from_disk(dataset_target_path)
    train_dataset = processed_splits["train"]
    val_dataset = processed_splits["test"]
    
    # Construct master config metadata profile for the tracking trainer core
    config = {
        "model_type": args.model, "seq_len": args.seq_len, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "lr": args.lr, "max_steps": args.max_steps,
        "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps,
        "run_name": run_name
    }
    
    # ==================================================================================
    # SAFE KWARGS EXTRACTION & ARCHITECTURAL ROUTING FACTORY
    # ==================================================================================
    model_kwargs = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_grad_checkpointing": args.grad_checkpointing
    }
    
    if args.model == "cs_lrad":
        model_kwargs["rank"] = args.rank
        model_kwargs["chunk_size"] = args.chunk_size  
    elif args.model in ["ccrs", "ckts", "cbkc", "cofe"]:
        model_kwargs["chunk_size"] = args.chunk_size  
    elif args.model == "transformer":
        model_kwargs["max_seq_len"] = max(args.seq_len, 4096)
        
    # Instantiate the safe architecture block straight out of the factory registry
    model = make_model(args.model, **model_kwargs).to(device)
    
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if global_rank == 0:
        print(f"==================================================================================")
        print(f"LAUNCHING BARETORCH ENGINE FACTORY TASK")
        print(f"TARGET ARCHITECTURE     : {args.model.upper()}")
        print(f"GRAD CHECKPOINTING      : {args.grad_checkpointing}")
        print(f"TOTAL ACTIVE PARAMETERS : {total_trainable_params:,}")
        print(f"MODEL DIMENSION PROFILE : d_model={args.d_model} | heads={args.num_heads} | layers={args.num_layers}")
        print(f"CONTEXT MESH CONFIG     : len={args.seq_len} | batch={args.batch_size} | accum={args.grad_accum}")
        print(f"==================================================================================")
        print(f"{'Step':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val PPL':<12} | {'LR':<10} | {'Accum Time':<12} | {'Tokens/Sec':<12}")
        print("-" * 115)
        
    dist.barrier()
    
    # Hand execution over to the centralized trainer subsystem
    trainer = CausalTrainer(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset,
        config=config, local_rank=local_rank, global_rank=global_rank, world_size=world_size
    )
    
    trainer.train()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()