import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def merge_evaluation_data():
    results_dir = "results"
    master_csv_path = os.path.join(results_dir, "master_metrics.csv")
    master_txt_path = os.path.join(results_dir, "master_inference_stats.txt")
    
    print("=" * 90)
    print("RUNNING BARETORCH RESULTS COLLATION & PLOTTING SUBSYSTEM")
    print("=" * 90)

    # -----------------------------------------------------------------
    # STEP 1: CONSOLIDATE TRAINING METRICS (CSVs)
    # -----------------------------------------------------------------
    csv_files = glob.glob(os.path.join(results_dir, "*/metrics.csv"))
    all_dfs = []
    
    print(f"\n[1/3] Scanning results tree... Found {len(csv_files)} training metric logs.")
    
    for csv_path in csv_files:
        folder_name = os.path.basename(os.path.dirname(csv_path))
        
        try:
            parts = folder_name.split("_")
            model_name = parts[0]
            if len(parts) == 4:
                model_name = f"{parts[0]}_{parts[1]}"
                seq_len = int(parts[3])
            else:
                seq_len = int(parts[2])
                
            df = pd.read_csv(csv_path)
            
            # --- BIT-PERFECT COLUMN SANITIZATION ---
            df.columns = df.columns.str.strip()
            
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'step':
                    rename_map[col] = 'Step'
                elif col_lower in ['train_loss', 'train loss']:
                    rename_map[col] = 'Train Loss'
                elif col_lower in ['train_tokens_per_sec', 'tokens/sec', 'tokens_sec']:
                    rename_map[col] = 'Tokens/Sec'
            df = df.rename(columns=rename_map)
            # ----------------------------------------------
            
            # Inject metadata identifiers directly into rows
            df["run_id"] = folder_name
            df["model"] = model_name.upper()
            df["seq_len"] = seq_len
            
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping broken or parsing-mismatched folder layout '{folder_name}': {e}")

    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv(master_csv_path, index=False)
        print(f"✅ Master metrics spreadsheet compiled successfully: {master_csv_path}")
    else:
        print("❌ No valid training metrics found.")
        master_df = None

    # -----------------------------------------------------------------
    # STEP 2: STITCH TOGETHER INFERENCE STATS (TXTs)
    # -----------------------------------------------------------------
    txt_files = glob.glob(os.path.join(results_dir, "*/inference_stats.txt"))
    print(f"\n[2/3] Collecting auto-regressive generation reports... Found {len(txt_files)} logs.")
    
    with open(master_txt_path, "w", encoding="utf-8") as master_out:
        for txt_path in sorted(txt_files):
            folder_name = os.path.basename(os.path.dirname(txt_path))
            
            master_out.write(f"\n" + "="*80 + f"\n")
            master_out.write(f" MASTER EXPERIMENT LEDGER TARGET: {folder_name.upper()}\n")
            master_out.write(f"="*80 + f"\n\n")
            
            with open(txt_path, "r", encoding="utf-8") as f_in:
                master_out.write(f_in.read())
            master_out.write("\n\n")
            
    print(f"✅ Master text validation ledger stitched successfully: {master_txt_path}")

    # -----------------------------------------------------------------
    # STEP 3: GENERATE GRAPHICAL PAPER FIGURES (PLOTS)
    # -----------------------------------------------------------------
    if master_df is not None:
        print(f"\n[3/3] Generating publication-grade Seaborn figures...")
        sns.set_theme(style="whitegrid")
        
        # Figure 1: Convergence Curves grouped by context length windows
        unique_lengths = master_df["seq_len"].unique()
        for length in unique_lengths:
            plt.figure(figsize=(9, 5.5))
            subset = master_df[master_df["seq_len"] == length]
            
            # Draw line curves tracking Loss across step checkpoints
            sns.lineplot(data=subset, x="Step", y="Train Loss", hue="model", linewidth=2.0)
            
            plt.title(f"Pre-training Convergence Trajectory (Context Sequence Length: {length})", fontsize=12, fontweight="bold", pad=12)
            plt.xlabel("Optimization Step Iteration", fontsize=10)
            plt.ylabel("Cross-Entropy Loss Value", fontsize=10)
            plt.legend(title="Architectures", loc="upper right")
            plt.tight_layout()
            
            plot_out = os.path.join(results_dir, f"convergence_curves_seq_{length}.png")
            plt.savefig(plot_out, dpi=300)
            plt.close()
            print(f"📈 Saved line curve plot to: {plot_out}")

        # Figure 2: System Throughput Efficiency Comparison Bar Chart
        final_steps = master_df.sort_values("Step").groupby("run_id").last().reset_index()
        
        if "Tokens/Sec" in final_steps.columns:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=final_steps, x="model", y="Tokens/Sec", hue="seq_len", palette="muted")
            
            plt.title("Hardware Processing Throughput vs Context Sequence Length Window", fontsize=12, fontweight="bold", pad=12)
            plt.xlabel("Evaluated Target Model Architecture", fontsize=10)
            plt.ylabel("Throughput Efficiency (Tokens per Second)", fontsize=10)
            plt.legend(title="Sequence Lengths")
            plt.tight_layout()
            
            bar_out = os.path.join(results_dir, "throughput_efficiency_matrix.png")
            plt.savefig(bar_out, dpi=300)
            plt.close()
            print(f"📊 Saved bar chart efficiency plot to: {bar_out}")
            
    print("\n" + "="*90)
    print("SUCCESS: Postprocessing cycle complete. Ready for paper manuscript compilation.")
    print("="*90)

if __name__ == "__main__":
    merge_evaluation_data()