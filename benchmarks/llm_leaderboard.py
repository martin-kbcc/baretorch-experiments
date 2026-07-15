import os
import sys
import argparse
import subprocess

def parse_leaderboard_args():
    parser = argparse.ArgumentParser(description="BareTorch Consolidated Downstream Leaderboard Execution Suite")
    
    # Execution Environment Configuration Pass-throughs
    parser.add_argument("--batch_size", type=int, default=128, help="Parallel sequence execution batch configuration allocation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target processing device slot mapping assignment")
    parser.add_argument("--limit", type=int, default=None, help="Debug tool: limit number of evaluation examples per task split for fast testing")
    
    return parser.parse_args()

def main():
    args = parse_leaderboard_args()
    
    # 1. Define the exact matrix array tracking your 2048 trained models checkpoint registry
    model_matrix = [
        {"name": "transformer", "heads": 4,  "ckpt": "checkpoints/transformer_tiny_2048.pt"},
        {"name": "gla",         "heads": 4,  "ckpt": "checkpoints/gla_tiny_2048.pt"},
        {"name": "gdn1",        "heads": 4,  "ckpt": "checkpoints/gdn1_tiny_2048.pt"},
        {"name": "gdn2",        "heads": 4,  "ckpt": "checkpoints/gdn2_tiny_2048.pt"},
        {"name": "mamba3",      "heads": 4,  "ckpt": "checkpoints/mamba3_tiny_2048.pt"},
        {"name": "cs_lrad",     "heads": 16, "ckpt": "checkpoints/cs_lrad_tiny_2048.pt"},
        {"name": "ccrs",        "heads": 16, "ckpt": "checkpoints/ccrs_tiny_2048.pt"},
    ]
    
    # Establish tracking indicators for clean visual parsing
    total_runs = len(model_matrix)
    
    print("=" * 95)
    print("STARTING BARETORCH CONSOLIDATED LLM DOWNSTREAM LEADERBOARD SWEEP")
    print(f"TARGET HARDWARE SLOT : {args.device.upper()}")
    print(f"EVALUATION PROFILE   : Task=lambada_openai | Batch={args.batch_size} | Limit={args.limit}")
    print(f"TOTAL REGISTERED RUNS: {total_runs} Models Scheduled")
    print("=" * 95)
    
    # Verify execution environment positioning context to prevent path mapping errors
    if not os.path.exists("runs/run_eval.py"):
        print("\n🚨 Error: Evaluation runner not found. Please execute this script from the project root directory.")
        print("Usage: PYTHONPATH=. python benchmarks/llm_leaderboard.py\n")
        sys.exit(1)

    # 2. Iterate through the model configurations array matrix sequentially
    for idx, model_cfg in enumerate(model_matrix, start=1):
        m_name = model_cfg["name"]
        m_heads = model_cfg["heads"]
        m_ckpt = model_cfg["ckpt"]
        
        print("\n" + "-" * 95)
        print(f"🏃 RUN [{idx}/{total_runs}]: Evaluating {m_name.upper()} (Heads={m_heads})")
        print(f"CHECKPOINT TARGET  : {m_ckpt}")
        print("-" * 95)
        
        # Verify checkpoint file exists on disk before launching subprocess shell
        if not os.path.exists(m_ckpt):
            print(f"⚠️ Warning: Checkpoint path '{m_ckpt}' not found. Skipping validation track.")
            continue
            
        # 3. Formulate the explicit command array arguments to feed the subprocess pipeline
        cmd = [
            "python", "runs/run_eval.py",
            "--model", str(m_name),
            "--checkpoint", str(m_ckpt),
            "--seq_len", "2048",
            "--num_heads", str(m_heads),
            "--tokens_trained", "4.2",
            "--batch_size", str(args.batch_size),
            "--tasks", "lambada_openai", # Firmly locked to our optimized metric lane
            "--device", str(args.device)
        ]
        
        # Append limit parameter array block only if explicitly requested via CLI
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
            
        # 4. Invoke the programmatic evaluation runtime with live stdout streaming
        try:
            # Set environment context explicitly to preserve tracking paths
            env = os.environ.copy()
            if "PYTHONPATH" not in env:
                env["PYTHONPATH"] = "."
            else:
                env["PYTHONPATH"] = f".:{env['PYTHONPATH']}"
                
            # Execute and wait for process cleanup closure before proceeding to next loop row
            result = subprocess.run(cmd, env=env, check=True)
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: Run complete for {m_name.upper()}. Master ledger appended.")
            else:
                print(f"❌ ERROR: Subprocess loop evaluated non-zero return code for model {m_name.upper()}")
                
        except subprocess.CalledProcessError as e:
            print(f"💥 CRITICAL: Subprocess execution failed violently for model {m_name.upper()}")
            print(f"Command tracing log: {' '.join(cmd)}")
            print(f"Error tracing message: {str(e)}")
            continue
            
    print("\n" + "=" * 95)
    print("🏆 SUCCESS: CONSOLIDATED LLM DOWNSTREAM LEADERBOARD SWEEP COMPLETE")
    print("All collected metrics are securely compiled inside: results/master_downstream_results.csv")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()