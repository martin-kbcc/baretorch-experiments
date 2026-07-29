import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

# ==============================================================================
#  1. Chunk-Segmented Competitive Associative Engine (CS-CANN)
# ==============================================================================

class CompetitiveAssociativeEngine(nn.Module):
    """
    Chunk-Segmented Competitive Associative Network (CS-CANN)
    
    Combines local exact Causal Softmax attention (C=32) with a dual-track 
    normalizing global state space (S_V value matrix + S_Z partition vector)
    to enforce scale-invariant memory retrieval in 100% pure PyTorch GEMMs.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Linear Projections
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Continuous Decay Gate (g) & Local/Global Interpolation Gate (alpha)
        self.W_gate  = nn.Linear(d_model, num_heads, bias=True)
        self.W_alpha = nn.Linear(d_model, self.inner_dim, bias=True)
        
        # Swish Output Gate & Final Linear Layer
        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        N = L // C  # Macro-chunks count
        scaling = 1.0 / math.sqrt(d_h)
        
        # Raw Projections: [B, H, N, C, d_h]
        Q_raw = self.W_q(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        K_raw = self.W_k(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        V     = self.W_v(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        # Strictly Positive Convex Feature Maps for Normalizing Memory: phi(x) > 0
        Q_pos = F.elu(Q_raw) + 1.0
        K_pos = F.elu(K_raw) + 1.0
        
        # --- 1. Intra-Chunk Exact Causal Softmax Pass ---
        scores = torch.matmul(Q_raw, K_raw.transpose(-1, -2)) * scaling
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        A_local = F.softmax(scores, dim=-1)
        Y_local = torch.matmul(A_local, V)
        
        # --- 2. Inter-Chunk Dual-Track Normalizing Memory Track ---
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        log_gate = torch.log(gate)
        Lambda = torch.cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(Lambda)
        
        chunk_decay_log = torch.sum(log_gate, dim=-2).squeeze(-1) 
        Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)  
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        M_chunks = torch.exp(log_M_chunks.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, -1e4))
        
        # Decay keys across macro-chunk boundaries
        K_decayed = K_pos * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        
        # Value State (S_V) and Normalizer Partition Vector State (S_Z)
        S_V = torch.matmul(M_chunks, torch.matmul(K_decayed.transpose(-1, -2), V).view(B, H, N, d_h * d_h)).view(B, H, N, d_h, d_h)
        
        K_sum_chunk = torch.sum(K_decayed, dim=-2)  # [B, H, N, d_h]
        S_Z = torch.matmul(M_chunks, K_sum_chunk)  # Shape: [B, H, N, d_h]
        
        # Normalized Expectation Retrieval
        Q_exp = Q_pos * exp_Lambda
        Y_num = torch.matmul(Q_exp, S_V) * scaling  # [B, H, N, C, d_h]
        Z_den = torch.sum(Q_exp * S_Z.unsqueeze(-2), dim=-1, keepdim=True) * scaling  # [B, H, N, C, 1]
        
        Y_global = Y_num / (Z_den + 1e-5)
        
        # --- 3. Convex Gated Local/Global Fusion ---
        alpha = torch.sigmoid(self.W_alpha(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        Y_fused = alpha * Y_local + (1.0 - alpha) * Y_global
        
        # Output Aggregation & Swish Gate
        Out = Y_fused.permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out * F.silu(self.W_swish_gate(x)))

    def step_inference(self, x, past_states=None):
        """
        Step-wise autoregressive decoding pass (L=1).
        """
        B, L, D = x.shape
        H, d_h = self.num_heads, self.d_head
        scaling = 1.0 / math.sqrt(d_h)
        
        Q_raw = self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        K_raw = self.W_k(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        V     = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        
        Q_pos = F.elu(Q_raw) + 1.0
        K_pos = F.elu(K_raw) + 1.0
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        
        if past_states is None:
            S_V_past = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
            S_Z_past = torch.zeros(B, H, d_h, device=x.device, dtype=x.dtype)
        else:
            S_V_past, S_Z_past = past_states
            
        S_V_state = gate * S_V_past + torch.matmul(K_pos.transpose(-1, -2), V)
        S_Z_state = gate * S_Z_past + K_pos.squeeze(-2)
        
        Y_local = V  # Single step local context
        
        Y_num = torch.matmul(Q_pos, S_V_state) * scaling
        Z_den = torch.sum(Q_pos * S_Z_state.unsqueeze(-2), dim=-1, keepdim=True) * scaling
        Y_global = Y_num / (Z_den + 1e-5)
        
        alpha = torch.sigmoid(self.W_alpha(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        Y_fused = alpha * Y_local + (1.0 - alpha) * Y_global
        
        return self.W_out(Y_fused.permute(0, 2, 1, 3).contiguous().view(B, L, D) * F.silu(self.W_swish_gate(x))), (S_V_state, S_Z_state)


# ==============================================================================
#  2. Pre-Norm Decoder Block Wrapping
# ==============================================================================

class CANNDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = CompetitiveAssociativeEngine(d_model, num_heads, chunk_size=chunk_size)
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

class CausalCANNLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CANNDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
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