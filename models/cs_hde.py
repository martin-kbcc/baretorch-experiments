import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

# ==============================================================================
#  1. Key-Guided Householder Delta Engine (CS-HDE v2)
# ==============================================================================

class HouseholderDeltaEngine(nn.Module):
    """
    Chunk-Segmented Householder Delta Engine (CS-HDE) - Key-Guided Variant
    
    Replaces unconstrained Householder reflections with Key-Anchored reflections 
    (U = K + W_u X) and modulates reflection intensity using Key-Value error tension,
    closing the perplexity gap with GatedDeltaNet2 while maintaining pure GEMM throughput.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Standard Linear Projections
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Key-Residual Reflection Projection (Learned shift off the Key vector)
        self.W_u = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Continuous Decay Gate (g) and Householder Intensity Gate (beta)
        self.W_gate = nn.Linear(d_model, num_heads, bias=True)        
        self.W_beta_gate = nn.Linear(d_model, num_heads, bias=True)  
        
        # Swish Output Gating & Final Linear Projection
        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

        # Soft Beta Initialization: Start near ~0.1 so the model acts as an identity
        # pass initially, learning to turn on Householder cleansing only when needed.
        nn.init.constant_(self.W_beta_gate.bias, -2.0)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        N = L // C  # Macro-chunks count
        
        # Projections & Reshaping to Macro-Chunks: [B, H, N, C, d_h]
        Q = F.silu(self.W_q(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        K = F.silu(self.W_k(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        V = self.W_v(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        # --- Upgrade 1: Key-Guided Reflection Direction (U = K + W_u X) ---
        U_proj = self.W_u(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        U = K + U_proj
        U_norm = F.normalize(U, p=2, dim=-1, eps=1e-6)
        
        # Continuous decay gate (g)
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        
        # --- Upgrade 2: Key-Value Error Scaling on Beta ---
        kv_tension = torch.sigmoid(torch.sum(K * V, dim=-1, keepdim=True))
        beta_gate = torch.sigmoid(self.W_beta_gate(x).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)) * kv_tension
        
        # Continuous log-domain decay tensor
        log_gate = torch.log(gate)
        Lambda = torch.cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(Lambda)  
        
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        
        # --- 1. Intra-Chunk Householder Memory Reflection ---
        # H(v) = v - 2 * beta * <u, v> * u
        proj_uv = torch.sum(U_norm * V, dim=-1, keepdim=True)
        V_cleansed = V - 2.0 * beta_gate * proj_uv * U_norm
        
        # Local decay-weighted causal attention map
        M_links = torch.exp((Lambda - Lambda.transpose(-1, -2)).masked_fill(causal_mask == 0, -1e4))
        scaling = 1.0 / math.sqrt(d_h)
        
        # Local sequence output pass
        Y_local = torch.matmul(torch.matmul(Q, K.transpose(-1, -2)) * scaling * M_links, V_cleansed)  
        
        # --- 2. Inter-Chunk Global Recurrence Track ---
        chunk_decay_log = torch.sum(log_gate, dim=-2).squeeze(-1) 
        Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)  
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        M_chunks = torch.exp(log_M_chunks.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, -1e4))
        
        # Decay keys across chunk boundaries and map historical states
        K_decayed = (K * beta_gate) * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        S_historical = torch.matmul(M_chunks, torch.matmul(K_decayed.transpose(-1, -2), V_cleansed).view(B, H, N, d_h * d_h)).view(B, H, N, d_h, d_h)
        
        Y_global = torch.matmul(Q * exp_Lambda, S_historical) * scaling  
        
        # Output Aggregation & Swish Gate
        Out = (Y_local + Y_global).permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out * F.silu(self.W_swish_gate(x)))

    def step_inference(self, x, past_S=None):
        """
        Step-wise autoregressive decoding pass (L=1).
        Performs O(d_h^2) Householder reflection on the historical memory matrix S.
        """
        B, L, D = x.shape
        H, d_h = self.num_heads, self.d_head
        scaling = 1.0 / math.sqrt(d_h)
        
        Q = F.silu(self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K = F.silu(self.W_k(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        V = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        
        # Key-Guided Alignment in step mode
        U_proj = self.W_u(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        U = K + U_proj
        U_norm = F.normalize(U, p=2, dim=-1, eps=1e-6)
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        
        kv_tension = torch.sigmoid(torch.sum(K * V, dim=-1, keepdim=True))
        beta_gate = torch.sigmoid(self.W_beta_gate(x).view(B, L, H).permute(0, 2, 1).unsqueeze(-1)) * kv_tension
        
        S_past = past_S if past_S is not None else torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
        
        # 1. Householder Memory Reflection on historical memory matrix S
        U_S_proj = torch.matmul(U_norm, S_past)
        S_reflected = torch.matmul(U_norm.transpose(-1, -2), U_S_proj)
        S_cleansed = S_past - 2.0 * beta_gate * S_reflected
        
        # 2. Continuous decay step
        S_decayed = gate * S_cleansed
        
        # 3. Additive value injection for current token
        S_state = S_decayed + torch.matmul(K.transpose(-1, -2), V)
        
        # 4. Generate step token representation
        Y_local = torch.matmul(Q, torch.matmul(K.transpose(-1, -2), V)) * scaling
        Y_global = torch.matmul(Q, S_decayed) * scaling
        Out = Y_local + Y_global
        
        return self.W_out(Out.permute(0, 2, 1, 3).contiguous().view(B, L, D) * F.silu(self.W_swish_gate(x))), S_state


# ==============================================================================
#  2. Pre-Norm Decoder Block Wrapping
# ==============================================================================

class HDEDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = HouseholderDeltaEngine(d_model, num_heads, chunk_size=chunk_size)
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

class CausalHDELM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HDEDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
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
            attn_out, state = block.attn.step_inference(h_attn, past_S=past_states[i])
            h = h + attn_out
            h = h + block.mlp(block.ln2(h))
            next_states.append(state)
            
        return self.lm_head(self.final_norm(h)), next_states