import os
import sys
import argparse
import datasets
import itertools
from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress verbose tokenizer sequence length tracking alerts
import transformers
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="BareTorch Dataset Tokenization & Chunk-Packing Subsystem")
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=2048, 
        help="Target auto-regressive context sequence length tracking boundary window"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Custom target directory path to serialize binary token arrays"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    seq_len = args.seq_len
    block_size = seq_len + 1  # Adding 1 token per window for exact causal tracking shifting
    
    # Auto-format unique cache directories if no custom layout path is explicitly declared
    if args.output_dir is not None:
        cache_dir = args.output_dir
    else:
        cache_dir = f"./processed_wiki_dataset_{seq_len}"
        
    if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "dataset_dict.json")):
        print(f"Preprocessed binary token array dataset already exists at '{cache_dir}'. Exiting pipeline.")
        print(f"NOTE: If you wish to recalculate the chunk alignments for seq_len={seq_len}, remove this folder manually.")
        sys.exit(0)
        
    print("=" * 85)
    print("LAUNCHING BARETORCH OFFLINE SPECTRAL DATA PREPROCESSING SUITE")
    print(f"TARGET SEQUENCE BINDING WINDOW : {seq_len} Tokens (Block Size: {block_size})")
    print(f"TARGET SERIALIZATION CHANNEL   : {cache_dir}")
    print("=" * 85)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("\n[1/7] Downloading master multilingual dataset to local storage cache...")
    raw_ds = load_dataset("HuggingFaceFW/clean-wikipedia", split="train", download_mode="force_redownload")
    
    print("\n[2/7] Filtering master rows for English sub-partition via vectorized Arrow matching...")
    raw_ds = raw_ds.filter(
        lambda x: [lang == 'eng' for lang in x], 
        batched=True, 
        input_columns=['iso639-3'], 
        num_proc=24
    )
    
    print("\n[3/7] Casting text feature columns to raw binary format to prevent decoding crashes...")
    raw_ds = raw_ds.cast_column("text", datasets.Value("binary"))
    
    print("\n[4/7] Slicing top 3,500,000 cleaned Wikipedia articles (~4B Token Runway)...")
    raw_ds = raw_ds.select(range(min(len(raw_ds), 3500000)))
    
    print("\n[5/7] Creating deterministic training (99.8%) and validation (0.2%) partitions...")
    ds_splits = raw_ds.train_test_split(test_size=0.002, seed=42)
    
    def tokenize_mapping_fn(examples):
        # Gracefully handle raw byte allocations by stripping out corrupt bytes seamlessly
        cleaned_texts = [
            t.decode('utf-8', errors='ignore') if isinstance(t, bytes) else t 
            for t in examples["text"]
        ]
        return tokenizer(cleaned_texts, add_special_tokens=False)
        
    def group_tokens_fn(examples):
        # Linear O(N) itertools chain to flatten token streams with absolute speed
        concatenated = {
            k: list(itertools.chain.from_iterable(examples[k])) 
            for k in examples.keys()
        }
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        return {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        
    print("\n[6/7] Distributing tokenization mapping across 32 Threadripper CPU cores...")
    tokenized_splits = ds_splits.map(tokenize_mapping_fn, batched=True, num_proc=24, remove_columns=raw_ds.column_names)
    
    print("\n[7/7] Packing token streams into continuous causal context sequence windows...")
    processed_splits = tokenized_splits.map(group_tokens_fn, batched=True, num_proc=24)
    
    print(f"\nWriting finalized binary array compilation files to disk: '{cache_dir}'...")
    processed_splits.save_to_disk(cache_dir)
    
    print("\n" + "=" * 85)
    print(f"SUCCESS: PREPROCESSING FOR SEQUENCE LENGTH {seq_len} COMPLETE.")
    print("=" * 85)

if __name__ == "__main__":
    main()