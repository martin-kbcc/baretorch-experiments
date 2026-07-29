import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

# ==============================================================================
#  1. Chunk-Segmented Adaptive Polynomial Expansion Engine (CS-APEX)
# ==============================================================================

class AdaptivePolynomialEngine(nn.Module):
    """
    Chunk-Segmented Adaptive Polynomial Expansion Engine (CS-APEX)
    
    Pairs local C=32 exact causal Softmax attention with a dual-track 
    polynomial state space (S_A, S_B) coupled via Hadamard resonance gates.
    Achieves maximum GEMM speed while providing high retrieval capacity.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Projections
        self.W_q  = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_ka = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_kb = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_va = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_vb = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Decay Gate & Resonant Coupling Gate
        self.W_gate  = nn.Linear(d_model, num_heads, bias=True)
        self.W_gamma = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Output Gating & Final Linear Layer
        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        N = L // C  # Macro-chunks count
        scaling = 1.0 / math.sqrt(d_h)
        
        # Reshape to Macro-Chunks: [B, H, N, C, d_h]
        Q  = F.silu(self.W_q(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        K_A = F.silu(self.W_ka(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        
        # Polynomial 2nd-Order Harmonic Key
        K_B = torch.tanh(self.W_kb(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)) * K_A
        
        V_A = self.W_va(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        V_B = self.W_vb(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        # --- 1. Intra-Chunk Exact Causal Softmax Pass ---
        scores = torch.matmul(Q, K_A.transpose(-1, -2)) * scaling
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        A_local = F.softmax(scores, dim=-1)
        Y_local = torch.matmul(A_local, V_A)
        
        # --- 2. Inter-Chunk Global Polynomial Recurrence Track ---
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        log_gate = torch.log(gate)
        Lambda = torch.cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(Lambda)
        
        chunk_decay_log = torch.sum(log_gate, dim=-2).squeeze(-1) 
        Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)  
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        M_chunks = torch.exp(log_M_chunks.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, -1e4))
        
        K_A_decayed = K_A * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        K_B_decayed = K_B * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        
        S_A = torch.matmul(M_chunks, torch.matmul(K_A_decayed.transpose(-1, -2), V_A).view(B, H, N, d_h * d_h)).view(B, H, N, d_h, d_h)
        S_B = torch.matmul(M_chunks, torch.matmul(K_B_decayed.transpose(-1, -2), V_B).view(B, H, N, d_h * d_h)).view(B, H, N, d_h, d_h)
        
        Y_A_global = torch.matmul(Q * exp_Lambda, S_A) * scaling
        Y_B_global = torch.matmul(Q * exp_Lambda, S_B) * scaling
        
        # --- 3. Cross-Harmonic Resonant Coupling ---
        gamma = torch.sigmoid(self.W_gamma(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        Y_global = Y_A_global + gamma * (Y_B_global * F.silu(Y_A_global))
        
        # Output Aggregation & Swish Gate
        Out = (Y_local + Y_global).permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out * F.silu(self.W_swish_gate(x)))

    def step_inference(self, x, past_S=None):
        """
        Step-wise autoregressive decoding pass (L=1).
        """
        B, L, D = x.shape
        H, d_h = self.num_heads, self.d_head
        scaling = 1.0 / math.sqrt(d_h)
        
        Q   = F.silu(self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K_A = F.silu(self.W_ka(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K_B = torch.tanh(self.W_kb(x).view(B, L, H, d_h).permute(0, 2, 1, 3)) * K_A
        V_A = self.W_va(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        V_B = self.W_vb(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        
        if past_S is None:
            S_A_past = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
            S_B_past = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
        else:
            S_A_past, S_B_past = past_S
            
        S_A_state = gate * S_A_past + torch.matmul(K_A.transpose(-1, -2), V_A)
        S_B_state = gate * S_B_past + torch.matmul(K_B.transpose(-1, -2), V_B)
        
        Y_local = torch.matmul(Q, torch.matmul(K_A.transpose(-1, -2), V_A)) * scaling
        
        Y_A_global = torch.matmul(Q, S_A_state) * scaling
        Y_B_global = torch.matmul(Q, S_B_state) * scaling
        
        gamma = torch.sigmoid(self.W_gamma(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        Y_global = Y_A_global + gamma * (Y_B_global * F.silu(Y_A_global))
        
        Out = Y_local + Y_global
        return self.W_out(Out.permute(0, 2, 1, 3).contiguous().view(B, L, D) * F.silu(self.W_swish_gate(x))), (S_A_state, S_B_state)


# ==============================================================================
#  2. Pre-Norm Decoder Block Wrapping
# ==============================================================================

class APEXDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = AdaptivePolynomialEngine(d_model, num_heads, chunk_size=chunk_size)
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

class CausalAPEXLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            APEXDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
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