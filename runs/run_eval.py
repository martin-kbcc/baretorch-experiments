import os
import sys
import json
import argparse
import pandas as pd
from pprint import pprint

# Dynamically inject the parent directory (project root) into Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lm_eval
from lm_eval.utils import handle_non_serializable
from eval.harness_wrapper import BareTorchEvalWrapper

def parse_eval_args():
    parser = argparse.ArgumentParser(description="BareTorch Downstream Benchmarking Execution Suite")
    
    # Target Model Setup & Weights Mapping
    parser.add_argument("--model", type=str, required=True, 
                        choices=["transformer", "gla", "gdn2", "mamba3", "cs_lrad", "ccrs", "ckts", "cbkc", "cofe"])
    parser.add_argument("--model_version", type=str, default="tiny", choices=["tiny", "small", "base"],
                        help="Model sizing tier variant scale allocation tracking category")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to target trained .pt checkpoint state dictionary")
    parser.add_argument("--tasks", type=str, default="lambada_openai,hellaswag,piqa,arc_challenge", help="Comma-separated tasks string to execute")
    
    # Model Architecture Passthroughs
    parser.add_argument("--d_model", type=int, default=256, help="Hidden embedding layer capacity dimension")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of tracking mixing heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Total number of cascaded decoder layer blocks")
    parser.add_argument("--chunk_size", type=int, default=32, help="Sequence chunk block partitioning length")
    parser.add_argument("--rank", type=int, default=8, help="Uniform bottleneck factor (r) for active delta engines")
    
    # Paper Ledger Metadata Arguments
    parser.add_argument("--seq_len", type=int, required=True, help="The target sequence context length the model was trained on")
    parser.add_argument("--tokens_trained", type=float, default=4.2, help="Total training tokens milestone consumed (expressed in Billions)")
    
    # Evaluation Engine Options
    parser.add_argument("--batch_size", type=int, default=32, help="Parallel sequence execution batch configuration allocation")
    parser.add_argument("--limit", type=int, default=None, help="Debug tool: limit number of evaluation examples per task split")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target processing device slot mapping assignment")
    
    return parser.parse_args()

def main():
    args = parse_eval_args()
    
    # 1. Clean parse out the target task string array
    task_list = [t.strip() for t in args.tasks.split(",")]
    
    # Calculate target output file system locations cleanly
    checkpoint_file_name = os.path.basename(args.checkpoint)
    run_uid = os.path.splitext(checkpoint_file_name)[0]
    output_directory = f"results/{run_uid}"
    os.makedirs(output_directory, exist_ok=True)
    summary_output_path = os.path.join(output_directory, "downstream_benchmarks.json")
    master_csv_path = "results/master_downstream_results.csv"

    print("=" * 95)
    print("INITIALIZING BARETORCH DOWNSTREAM BENCHMARK EVALUATOR")
    print(f"TARGET MODEL        : {args.model.upper()}")
    print(f"MODEL VERSION       : {args.model_version.upper()}")
    print(f"CHECKPOINT STREAM   : {args.checkpoint}")
    print(f"METADATA METRICS    : Context={args.seq_len} | Data Run={args.tokens_trained}B Tokens | Batch={args.batch_size}")
    print(f"EVALUATION PROFILE  : Tasks={task_list} | Device={args.device} | Limit={args.limit}")
    print("=" * 95)

    # 2. Extract configuration specific kwargs matching architectural signatures safely
    extra_model_kwargs = {}
    if args.model == "cs_lrad":
        extra_model_kwargs["rank"] = args.rank
        extra_model_kwargs["chunk_size"] = args.chunk_size
    elif args.model in ["ccrs", "ckts", "cbkc", "cofe"]:
        extra_model_kwargs["chunk_size"] = args.chunk_size

    # 3. Instantiate our custom registered evaluation harness bridge
    lm_eval_bridge = BareTorchEvalWrapper(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        device=args.device,
        batch_size=args.batch_size,
        **extra_model_kwargs
    )

    # Automatically calculate parameter metrics dynamically using framework layers
    total_params = 0
    if hasattr(lm_eval_bridge, "model"):
        total_params = sum(p.numel() for p in lm_eval_bridge.model.parameters() if p.requires_grad)
    param_count_m = round(total_params / 1e6, 2) if total_params > 0 else "N/A"

    print(f"\nCalculated Model Parameters: {param_count_m} Million")
    print("Handing processing over to EleutherAI LM-Eval-Harness Core Pipeline Engine...")
    
    # 4. Invoke the programmatic evaluation runtime
    eval_results = lm_eval.simple_evaluate(
        model=lm_eval_bridge,
        tasks=task_list,
        num_fewshot=0,  # Zero-shot validation tracking parameters
        limit=args.limit
    )

    # 5. Flush structural results out to localized JSON files for ledger tracking
    if eval_results is not None:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, default=handle_non_serializable, indent=2)
        
        print("\n" + "=" * 95)
        print("DOWNSTREAM EVALUATION ACCURACY METRIC PROFILE")
        print("-" * 95)
        print(f"{'Task Name':<25} | {'Metric Target':<25} | {'Value Output':<15}")
        print("-" * 95)
        
        flat_task_metrics = {}
        
        for task_name, task_metrics in eval_results.get("results", {}).items():
            for metric_name, value in task_metrics.items():
                if ",stderr" not in metric_name and "alias" not in metric_name:
                    try:
                        val_float = float(value)
                        val_str = f"{val_float:<15.4f}"
                        if "acc" in metric_name or "perplexity" in metric_name:
                            flat_task_metrics[f"{task_name}_{metric_name.split(',')[0]}"] = round(val_float, 4)
                    except (ValueError, TypeError):
                        val_str = f"{str(value):<15}"
                    
                    print(f"{task_name:<25} | {metric_name:<25} | {val_str}")
                    
        print("=" * 95)
        print(f"SUCCESS: Persistent validation record written to: {summary_output_path}")

        # -----------------------------------------------------------------
        # AUTOMATED MASTER LEDGER RECTIFICATION LOOP (CSV APPEND/UPDATE)
        # -----------------------------------------------------------------
        row_entry = {
            "Model Name": args.model.upper(),
            "Model Version": args.model_version.lower(),
            "Sequence Length": args.seq_len,
            "Parameters (M)": param_count_m,
            "Tokens Trained (B)": args.tokens_trained
        }
        
        for task_key, metric_val in flat_task_metrics.items():
            row_entry[task_key] = metric_val
            
        new_row_df = pd.DataFrame([row_entry])
        
        if os.path.exists(master_csv_path):
            try:
                master_df = pd.read_csv(master_csv_path)
                
                # FIXED: Match on Name, Version scale, and Context Length to prevent overwrite corruption
                match_condition = (
                    (master_df["Model Name"] == args.model.upper()) & 
                    (master_df["Model Version"] == args.model_version.lower()) &
                    (master_df["Sequence Length"] == args.seq_len)
                )
                if match_condition.any():
                    master_df = master_df[~match_condition]  
                master_df = pd.concat([master_df, new_row_df], ignore_index=True)
            except Exception:
                master_df = new_row_df
        else:
            master_df = new_row_df
            
        master_df.to_csv(master_csv_path, index=False)
        print(f"📋 Master evaluation matrix logged successfully at: {master_csv_path}\n")
    else:
        print("\n🚨 Evaluation failed: Pipeline engine returned null results profile mapping.")

if __name__ == "__main__":
    main()