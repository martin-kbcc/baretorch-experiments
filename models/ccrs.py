import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

class ChunkwiseCascadedResonanceSifter(nn.Module):
    """
    Chunkwise Cascaded Resonance Sifter (CCRS).
    Implements a second-order recurrent filtering grid by cascading two distinct
    internal state chambers. Unrolls the coupled differential system globally in parallel 
    via lower-triangular matrix operator composition. Strictly O(L) memory footprint.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Dual-Chamber Second-Order Resonance Control Gates
        self.W_gate1 = nn.Linear(d_model, num_heads, bias=True)  # Chamber 1: Velocity/Damping Field
        self.W_gate2 = nn.Linear(d_model, num_heads, bias=True)  # Chamber 2: Position/Stiffness Field
        
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        C = self.chunk_size
        d_h = self.d_head
        N = L // C  
        
        Q = self.W_q(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        K = self.W_k(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        V = self.W_v(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        Q = F.silu(Q)
        K = F.silu(K)
        
        g1 = torch.clamp(torch.sigmoid(self.W_gate1(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        g2 = torch.clamp(torch.sigmoid(self.W_gate2(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        
        log_g1 = torch.log(g1)
        log_g2 = torch.log(g2)
        
        Lambda1 = torch.cumsum(log_g1, dim=-2)
        Lambda2 = torch.cumsum(log_g2, dim=-2)
        
        log_M1 = Lambda1 - Lambda1.transpose(-1, -2)
        log_M2 = Lambda2 - Lambda2.transpose(-1, -2)
        
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        log_M1 = log_M1.masked_fill(causal_mask == 0, float('-inf'))
        log_M2 = log_M2.masked_fill(causal_mask == 0, float('-inf'))
        
        M1 = torch.exp(log_M1)
        M2 = torch.exp(log_M2)
        
        M_cascaded = torch.matmul(M2, M1)
        
        scaling = 1.0 / math.sqrt(d_h)
        A_local = torch.matmul(Q, K.transpose(-1, -2)) * scaling
        Y_local = torch.matmul(A_local * M_cascaded, V)
        
        chunk_decay1_log = torch.sum(log_g1, dim=-2).squeeze(-1) 
        chunk_decay2_log = torch.sum(log_g2, dim=-2).squeeze(-1) 
        
        K_decayed = K * torch.exp(Lambda1[:, :, :, -1:, :] - Lambda1)
        S_delta = torch.matmul(K_decayed.transpose(-1, -2), V) 
        
        Lambda_c1 = torch.cumsum(chunk_decay1_log, dim=2)
        Lambda_c2 = torch.cumsum(chunk_decay2_log, dim=2)
        
        log_Mc1 = Lambda_c1.unsqueeze(-1) - Lambda_c1.unsqueeze(-2) - chunk_decay1_log.unsqueeze(-1)
        log_Mc2 = Lambda_c2.unsqueeze(-1) - Lambda_c2.unsqueeze(-2) - chunk_decay2_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        log_Mc1 = log_Mc1.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, float('-inf'))
        log_Mc2 = log_Mc2.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, float('-inf'))
        
        Mc1 = torch.exp(log_Mc1)
        Mc2 = torch.exp(log_Mc2)
        
        Mc_cascaded = torch.matmul(Mc2, Mc1)
        
        S_delta_flat = S_delta.view(B, H, N, d_h * d_h)
        S_historical_flat = torch.matmul(Mc_cascaded, S_delta_flat)
        S_historical = S_historical_flat.view(B, H, N, d_h, d_h)
        
        Q_decayed = Q * torch.exp(Lambda2)
        Y_global = torch.matmul(Q_decayed, S_historical) * scaling
        
        Out = Y_local + Y_global
        Out = Out.permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out)

    def step_inference(self, x, past_S=None):
        B, L, D = x.shape
        H, d_h = self.num_heads, self.d_head
        
        Q = F.silu(self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K = F.silu(self.W_k(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        V = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        
        g1 = torch.clamp(torch.sigmoid(self.W_gate1(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        g2 = torch.clamp(torch.sigmoid(self.W_gate2(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        
        if past_S is None:
            S1 = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
            S2 = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
        else:
            S1, S2 = past_S
            
        scaling = 1.0 / math.sqrt(d_h)
        KV = torch.matmul(K.transpose(-1, -2), V) * scaling
        
        S1 = g1 * S1 + KV
        S2 = g2 * S2 + S1
        
        Y_step = torch.matmul(Q, S2)
        Out = Y_step.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.W_out(Out), (S1, S2)


class CCRSDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = ChunkwiseCascadedResonanceSifter(d_model, num_heads, chunk_size=chunk_size)
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


class CausalCCRSLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CCRSDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
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
            
        logits = self.lm_head(self.final_norm(h))
        return logits[:, :L, :]

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