import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

# ==============================================================================
#  1. Fully Vectorized Hyper-Differential Subspace Attention Engine (HDSA)
# ==============================================================================

class HyperDifferentialSubspaceEngine(nn.Module):
    """
    Hyper-Differential Subspace Attention Engine (HDSA)
    
    - Differential Softmax Attention Track: Full Unchunked Sequence L=2048 via Dual Flash-SDPA.
      Mathematically exact identity: Softmax(Q1 K1^T) V - lambda * Softmax(Q2 K2^T) V
      Zero intermediate L x L VRAM allocation (O(L) memory via native PyTorch FlashAttention).
    - Subspace Delta Recurrence Track: Chunk-Parallelized (C=32) Nilpotent Subspace Delta Engine.
    - Non-linear 2nd-Order Polynomial Value Path (V_poly).
    - SwiGLU Convex Output Fusion.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32, conv_kernel=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.conv_kernel = conv_kernel
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Dual-Stream Differential Projections (Q1, K1, Q2, K2)
        self.W_q1 = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k1 = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_q2 = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k2 = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Head-wise Differential Lambda parameter (initialized near 0.8)
        self.lambda_param = nn.Parameter(torch.ones(num_heads) * 0.8)
        
        # Higher-Order Polynomial Value Path
        self.W_v1 = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v2 = nn.Linear(d_model, self.inner_dim, bias=False)
        self.conv1d_v = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_kernel,
            groups=self.inner_dim,
            padding=conv_kernel - 1,
            bias=True
        )
        
        # Vectorized Subspace Delta Path
        self.W_q_delta = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k_delta = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_beta    = nn.Linear(d_model, self.inner_dim, bias=True)
        self.W_gate    = nn.Linear(d_model, self.num_heads, bias=True)
        
        # SwiGLU Output Fusion Gate
        self.W_fusion_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_swish_gate  = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out         = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        N = L // C  # Macro-chunks count
        scaling = 1.0 / math.sqrt(d_h)
        
        # --- 1. Polynomial Value Tensor V_poly ---
        V1 = self.W_v1(x)
        V2 = self.W_v2(x)
        V1_conv = self.conv1d_v(V1.transpose(1, 2))[:, :, :L].transpose(1, 2)
        V_poly_full = V1 * F.silu(V2) + V1_conv  # [B, L, inner_dim]
        V_poly_att  = V_poly_full.view(B, L, H, d_h).permute(0, 2, 1, 3).contiguous()  # [B, H, L, d_h]
        
        # --- 2. FULL UNCHUNKED Full-Sequence Differential Softmax via Dual Flash-SDPA ---
        Q1 = self.W_q1(x).view(B, L, H, d_h).permute(0, 2, 1, 3).contiguous()
        K1 = self.W_k1(x).view(B, L, H, d_h).permute(0, 2, 1, 3).contiguous()
        Q2 = self.W_q2(x).view(B, L, H, d_h).permute(0, 2, 1, 3).contiguous()
        K2 = self.W_k2(x).view(B, L, H, d_h).permute(0, 2, 1, 3).contiguous()
        
        # Dual FlashAttention passes with zero L x L intermediate memory allocation
        Y1 = F.scaled_dot_product_attention(Q1, K1, V_poly_att, is_causal=True)
        Y2 = F.scaled_dot_product_attention(Q2, K2, V_poly_att, is_causal=True)
        
        lam = torch.sigmoid(self.lambda_param).view(1, H, 1, 1)
        Y_diff = Y1 - lam * Y2  # [B, H, L, d_h]
        
        # --- 3. Vectorized Subspace Delta Recurrence Track (C=32) ---
        V_poly_chunk = V_poly_full.view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        Q_delta = F.silu(self.W_q_delta(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        K_delta = F.silu(self.W_k_delta(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        beta    = torch.sigmoid(self.W_beta(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        gate    = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        
        # Intra-Chunk Nilpotent Delta Cleansing
        K_beta = K_delta * beta
        G = torch.matmul(K_beta, K_delta.transpose(-1, -2))
        causal_mask_strict = torch.tril(torch.ones(C, C, device=x.device), diagonal=-1).view(1, 1, 1, C, C)
        T = G * causal_mask_strict
        
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T2 = torch.matmul(T, T)
        T3 = torch.matmul(T2, T)
        M_delta = I_mat - T + T2 - T3
        V_clean = torch.matmul(M_delta, V_poly_chunk)
        
        # Local & Inter-Chunk Global Delta Output
        causal_mask_chunk = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        A_local_delta = (torch.matmul(Q_delta, K_delta.transpose(-1, -2)) * scaling).masked_fill(causal_mask_chunk == 0, 0.0)
        Y_local_delta = torch.matmul(A_local_delta, V_clean)
        
        log_gate = torch.log(gate)
        Lambda = torch.cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(Lambda)
        
        chunk_decay_log = torch.sum(log_gate, dim=-2).squeeze(-1) 
        Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)  
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        M_chunks = torch.exp(log_M_chunks.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, -1e4))
        
        K_decayed = K_delta * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        S_historical = torch.matmul(M_chunks, torch.matmul(K_decayed.transpose(-1, -2), V_clean).view(B, H, N, d_h * d_h)).view(B, H, N, d_h, d_h)
        Y_global_delta = torch.matmul(Q_delta * exp_Lambda, S_historical) * scaling  
        
        Y_delta = (Y_local_delta + Y_global_delta).permute(0, 2, 3, 1, 4).contiguous().view(B, H, L, d_h)
        
        # --- 4. SwiGLU Convex Fusion ---
        fusion_gate = torch.sigmoid(self.W_fusion_gate(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        Y_fused = fusion_gate * Y_diff + (1.0 - fusion_gate) * Y_delta
        
        Out = Y_fused.permute(0, 2, 1, 3).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out * F.silu(self.W_swish_gate(x)))

    def step_inference(self, x, past_states=None):
        B, L, D = x.shape
        H, d_h = self.num_heads, self.d_head
        scaling = 1.0 / math.sqrt(d_h)
        
        if past_states is None:
            conv_buffer = torch.zeros(B, self.inner_dim, self.conv_kernel - 1, device=x.device, dtype=x.dtype)
            S_past = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
        else:
            conv_buffer, S_past = past_states
            
        V1 = self.W_v1(x)
        V2 = self.W_v2(x)
        v1_cat = torch.cat([conv_buffer, V1.transpose(1, 2)], dim=-1)
        next_conv_buffer = v1_cat[:, :, 1:]
        V1_conv = F.conv1d(v1_cat, self.conv1d_v.weight, self.conv1d_v.bias, groups=self.inner_dim).transpose(1, 2)
        V_poly = (V1 * F.silu(V2) + V1_conv).view(B, L, H, d_h).permute(0, 2, 1, 3)
        
        Q1 = self.W_q1(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        K1 = self.W_k1(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        Q2 = self.W_q2(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        K2 = self.W_k2(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        
        scores1 = torch.matmul(Q1, K1.transpose(-1, -2)) * scaling
        scores2 = torch.matmul(Q2, K2.transpose(-1, -2)) * scaling
        A1 = F.softmax(scores1, dim=-1)
        A2 = F.softmax(scores2, dim=-1)
        lam = torch.sigmoid(self.lambda_param).view(1, H, 1, 1)
        A_diff = A1 - lam * A2
        Y_diff = torch.matmul(A_diff, V_poly)
        
        Q_delta = F.silu(self.W_q_delta(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K_delta = F.silu(self.W_k_delta(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        beta    = torch.sigmoid(self.W_beta(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        gate    = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, L, H, 1).permute(0, 2, 1, 3), min=1e-3, max=0.999)
        
        S_retrieved = torch.matmul(K_delta, S_past)
        V_clean = V_poly - beta * S_retrieved
        S_state = gate * S_past + torch.matmul(K_delta.transpose(-1, -2), V_clean)
        
        Y_local = torch.matmul(Q_delta, torch.matmul(K_delta.transpose(-1, -2), V_clean)) * scaling
        Y_global = torch.matmul(Q_delta, S_past) * scaling
        Y_delta = Y_local + Y_global
        
        fusion_gate = torch.sigmoid(self.W_fusion_gate(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        Y_fused = fusion_gate * Y_diff + (1.0 - fusion_gate) * Y_delta
        
        return self.W_out(Y_fused.permute(0, 2, 1, 3).contiguous().view(B, L, D) * F.silu(self.W_swish_gate(x))), (next_conv_buffer, S_state)


# ==============================================================================
#  2. Pre-Norm Decoder Block Wrapping
# ==============================================================================

class HDSADecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = HyperDifferentialSubspaceEngine(d_model, num_heads, chunk_size=chunk_size)
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


# ==============================================================================
#  3. Full Causal Language Model
# ==============================================================================

class CausalHDSALM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HDSADecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        chunk_size = self.layers[0].attn.chunk_size
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        if pad_len > 0: 
            x = F.pad(x, (0, pad_len), value=0)
            
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h))[:, :L, :]

    def step_inference(self, x, past_states=None):
        h = self.token_embedding(x)
        next_states = []
        if past_states is None: 
            past_states = [None] * len(self.layers)
            
        for i, block in enumerate(self.layers):
            h_attn = block.ln1(h)
            attn_out, state = block.attn.step_inference(h_attn, past_states=past_states[i])
            h = h + attn_out
            h = h + block.mlp(block.ln2(h))
            next_states.append(state)
            
        return self.lm_head(self.final_norm(h)), next_states