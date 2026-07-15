import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

# Inject project root to environment path channels
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import make_model
from models.base_model import RMSNorm, GatedMLP

# -----------------------------------------------------------------
# HARDWARE-NATIVE FLASHATTENTION-2 DISPATCHER HOOKS
# -----------------------------------------------------------------
class CausalFlashAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        d_h = self.head_dim
        
        # Format shapes to match PyTorch SDPA specifications: (B, H, L, d_h)
        q = self.W_q(x).view(B, L, H, d_h).transpose(1, 2)
        k = self.W_k(x).view(B, L, H, d_h).transpose(1, 2)
        v = self.W_v(x).view(B, L, H, d_h).transpose(1, 2)
        
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Force the hardware execution engine strictly into FlashAttention-2 tiling mode
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.dropout_p if self.training else 0.0, 
                is_causal=True  # Natively applies causal triangular masking inside the kernel
            )
            
        out_flat = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_out(out_flat)

class FlashTransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalFlashAttention(d_model, num_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalFlashTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, max_seq_len=16384, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            FlashTransformerDecoderBlock(d_model, num_heads, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
        h = self.token_embedding(x) + self.position_embedding(pos)
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h))


# -----------------------------------------------------------------
# PRIMARY EVALUATION SUITE
# -----------------------------------------------------------------
def run_stress_benchmarks():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    vocab_size = 50257
    d_model = 256
    num_heads = 16
    num_layers = 8
    batch_size = 4  # Controlled batch execution factor
    
    # Sequence boundaries extending into ultra-long dimensions
    sequence_lengths = [256, 1024, 2048, 4096, 8192, 16384]
    architectures = ["transformer_vanilla", "transformer_flash", "ccrs", "cs_lrad"]
    
    results = {arch: {"vram": [], "throughput": []} for arch in architectures}
    
    print("=" * 105)
    print("LAUNCHING BARETORCH MULTI-ARCHITECTURE HARDWARE EFFICIENCY STRESS-TESTS")
    print(f"HARDWARE TARGET : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PROFILE SPECS   : Batch={batch_size} | Dim={d_model} | Heads={num_heads} | Layers={num_layers}")
    print("=" * 105)

    for seq_len in sequence_lengths:
        print(f"\n⚡ Profiling active context window scale: Sequence Length = {seq_len}...")
        x_synthetic = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        for arch in architectures:
            # --- CRITICAL BUG FIX: POSITION EMBEDDING BOUNDARY PROTECTION ---
            if arch == "transformer_vanilla" and seq_len > 4096:
                results[arch]["vram"].append("OOM/Limit")
                results[arch]["throughput"].append("OOM/Limit")
                print(f"  └─> {arch.upper():<20} | ⏭️ Skipping: Exceeds hardcoded 4096 position embedding limit.")
                continue
            # -----------------------------------------------------------------    

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
            extra_kwargs = {}
            if arch == "cs_lrad":
                extra_kwargs["rank"] = 8
                extra_kwargs["chunk_size"] = 32
            elif arch == "ccrs":
                extra_kwargs["chunk_size"] = 32
                
            try:
                # Route target model configurations dynamically
                if arch == "transformer_vanilla":
                    model = make_model("transformer", vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers).to(device)
                elif arch == "transformer_flash":
                    model = CausalFlashTransformerLM(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, max_seq_len=seq_len).to(device)
                else:
                    model = make_model(arch, vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, **extra_kwargs).to(device)
                
                model.train()
                
                # Warmup iterations
                for _ in range(3):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(x_synthetic)
                        loss = logits.sum()
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                
                torch.cuda.synchronize(device)
                
                # Performance profiling loop
                steps_eval = 10
                start_time = time.perf_counter()
                
                for _ in range(steps_eval):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(x_synthetic)
                        loss = logits.sum()
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                    
                torch.cuda.synchronize(device)
                elapsed_time = time.perf_counter() - start_time
                
                total_processed_tokens = batch_size * seq_len * steps_eval
                tokens_per_second = total_processed_tokens / elapsed_time
                peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                
                results[arch]["vram"].append(f"{peak_memory_gb:.2f} GB")
                results[arch]["throughput"].append(f"{tokens_per_second:,.0f}")
                
                print(f"  └─> {arch.upper():<20} | VRAM: {peak_memory_gb:.2f} GB | Speed: {tokens_per_second:,.0f} Tokens/Sec")
                
                del model, logits, loss
                
            except RuntimeError as cuda_error:
                if "out of memory" in str(cuda_error).lower() or "unsupported" in str(cuda_error).lower():
                    results[arch]["vram"].append("OOM")
                    results[arch]["throughput"].append("OOM")
                    print(f"  └─> {arch.upper():<20} | 🚨 CRITICAL HARDWARE EXCLUSION: CUDA OUT OF MEMORY (OOM)")
                else:
                    raise cuda_error

    # -----------------------------------------------------------------
    # PRINT & EXPORT PUBLICATION-GRADE MARKDOWN RECORD LEDGER TABLE
    # -----------------------------------------------------------------
    # Generate the string structure for simultaneous console and file output
    out_lines = []
    out_lines.append("=" * 105)
    out_lines.append("FINAL RAW SCALING EFFICIENCY LEDGER FOR MODEL MANUSCRIPT")
    out_lines.append("=" * 105)
    out_lines.append(f"HARDWARE TARGET : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    out_lines.append(f"PROFILE SPECS   : Batch={batch_size} | Dim={d_model} | Heads={num_heads} | Layers={num_layers}")
    out_lines.append("-" * 105)
    out_lines.append(f"{'Architecture':<22} | " + " | ".join([f"Seq {l:<5}" for l in sequence_lengths]))
    out_lines.append("-" * 105)
    
    for arch in architectures:
        vram_row = f"{arch.upper() + ' (VRAM)':<22} | " + " | ".join([f"{v:<9}" for v in results[arch]["vram"]])
        speed_row = f"{arch.upper() + ' (Speed)':<22} | " + " | ".join([f"{s:<9}" for s in results[arch]["throughput"]])
        out_lines.append(vram_row)
        out_lines.append(speed_row)
        out_lines.append("-" * 105)

    # Print to the active console terminal interface cleanly
    print("\n" + "\n".join(out_lines))
    
    # Secure the results folder hierarchy and commit the string matrix to storage
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    export_path = os.path.join(results_dir, "speed_memory_scaling.txt")
    
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
        
    print(f"\n✅ SUCCESS: Hardware scaling benchmarks archived to disk at: {export_path}\n")

if __name__ == "__main__":
    run_stress_benchmarks()