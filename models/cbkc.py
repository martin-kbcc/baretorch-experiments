import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from models.base_model import RMSNorm, GatedMLP

class CausalBlockKroneckerCascade(nn.Module):
    """
    Causal Block-Kronecker Cascade (CBKC).
    An inherently sub-quadratic sequence mixer operating entirely via pure GEMM blocks.
    Stabilized via hierarchical tier-dependent variance normalization.
    """
    def __init__(self, d_model=256, num_heads=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Core feature projection operators
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Multi-Scale Data-Dependent Kronecker Gating Network
        self.num_stages = 11  # Supports up to 2048 token sequence windows
        self.W_gates = nn.Linear(d_model, self.num_stages * self.num_heads, bias=True)
        
        self.q_norm = RMSNorm(self.inner_dim)
        self.k_norm = RMSNorm(self.inner_dim)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        d_h = self.d_head
        
        # Step 1: Project features into multi-head spaces
        Q = self.q_norm(self.W_q(x)).view(B, L, H, d_h).permute(0, 2, 1, 3) # (B, H, L, d_h)
        K = self.k_norm(self.W_k(x)).view(B, L, H, d_h).permute(0, 2, 1, 3) # (B, H, L, d_h)
        V = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)              # (B, H, L, d_h)
        
        # Extract data-dependent structural scale gates
        gates = torch.sigmoid(self.W_gates(x)).view(B, L, self.num_stages, H)
        gates = gates.permute(2, 0, 3, 1).unsqueeze(-1) # (num_stages, B, H, L, 1)
        
        # Initialize an independent accumulation tensor for the updates
        Y_accum = torch.zeros_like(V)
        
        # Step 2: Execute the Scale-Invariant Hierarchical Cascade
        for stage in range(self.num_stages):
            block_size = 2 ** (stage + 1)
            half_size = block_size // 2
            num_blocks = L // block_size
            
            if num_blocks == 0:
                break
                
            Q_b = Q.view(B, H, num_blocks, 2, half_size, d_h)
            K_b = K.view(B, H, num_blocks, 2, half_size, d_h)
            V_b = V.view(B, H, num_blocks, 2, half_size, d_h)
            
            K_past = K_b[:, :, :, 0]
            V_past = V_b[:, :, :, 0]
            Q_future = Q_b[:, :, :, 1]
            
            S_past = torch.matmul(K_past.transpose(-1, -2), V_past) 
            Y_future_update = torch.matmul(Q_future, S_past)
            
            scale_factor = 1.0 / math.sqrt(half_size * d_h)
            Y_future_update = Y_future_update * scale_factor
            
            stage_gate = gates[stage].view(B, H, num_blocks, 2, half_size, 1)
            Y_future_update = Y_future_update * stage_gate[:, :, :, 1]
            
            update_block = torch.zeros_like(Q_b)
            update_block[:, :, :, 1] = Y_future_update
            
            Y_accum = Y_accum + update_block.view(B, H, L, d_h)
            
        Y = V + (Y_accum * (1.0 / math.sqrt(self.num_stages)))
        
        Out = Y.permute(0, 2, 1, 3).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out)

    def step_inference(self, x, past_S=None):
        # (Inference implementation remains unchanged)
        B, L, D = x.shape
        H, d_h = self.num_heads, self.d_head
        Q = self.q_norm(self.W_q(x)).view(B, H, 1, d_h)
        K = self.k_norm(self.W_k(x)).view(B, H, 1, d_h)
        V = self.W_v(x).view(B, H, 1, d_h)
        gates = torch.sigmoid(self.W_gates(x)).view(B, L, self.num_stages, H).permute(2, 0, 3, 1).unsqueeze(-1)
        if past_S is None: K_cache, V_cache, step_idx = [], [], 0
        else: K_cache, V_cache, step_idx = past_S
        K_cache.append(K); V_cache.append(V)
        Y_accum = torch.zeros_like(V)
        for stage in range(self.num_stages):
            block_size = 2 ** (stage + 1); half_size = block_size // 2
            if (step_idx // half_size) % 2 == 1:
                start_idx = (step_idx // block_size) * block_size; end_idx = start_idx + half_size
                K_past = torch.cat(K_cache[start_idx:end_idx], dim=2)
                V_past = torch.cat(V_cache[start_idx:end_idx], dim=2)
                S_past = torch.matmul(K_past.transpose(-1, -2), V_past)
                Y_update = torch.matmul(Q, S_past) * (1.0 / math.sqrt(half_size * d_h)) * gates[stage]
                Y_accum = Y_accum + Y_update
        Y_token = V + (Y_accum * (1.0 / math.sqrt(self.num_stages)))
        step_idx += 1
        return self.W_out(Y_token.permute(0, 2, 1, 3).contiguous().view(B, L, self.inner_dim)), (K_cache, V_cache, step_idx)


class CBKCDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalBlockKroneckerCascade(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x):
        def _block_forward(x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
        
        if self.use_grad_checkpointing:
            return checkpoint.checkpoint(_block_forward, x, use_reentrant=False)
        else:
            return _block_forward(x)


class CausalCBKCLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CBKCDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h))

    def step_inference(self, x, past_states=None):
        h = self.token_embedding(x)
        next_states = []
        if past_states is None: past_states = [None] * len(self.layers)
        for i, block in enumerate(self.layers):
            h_attn = block.ln1(h)
            attn_out, state = block.attn.step_inference(h_attn, past_S=past_states[i])
            h = h + attn_out
            h = h + block.mlp(block.ln2(h))
            next_states.append(state)
        return self.lm_head(self.final_norm(h)), next_states