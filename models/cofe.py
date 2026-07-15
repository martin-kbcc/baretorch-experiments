import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

class CausalOrthogonalFeedbackEngine(nn.Module):
    """
    Causal Orthogonal Feedback Engine (COFE).
    Replicates the error-wiping performance of Gated Delta Net using an unrolled
    causal least-squares gradient projection step over chunked key matrices.
    Upgraded with non-linear feature projections to maximize parameter expressivity.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Flagship Projections
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Adaptive Error-Feedback Gate Generators
        self.W_beta = nn.Linear(d_model, self.num_heads, bias=True)
        self.W_alpha = nn.Linear(d_model, self.num_heads, bias=True)
        self.W_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        
        self.q_norm = RMSNorm(self.inner_dim)
        self.k_norm = RMSNorm(self.inner_dim)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        N = L // C
        
        # Step 1: Project inputs and isolate multi-head channels
        Q = F.silu(self.q_norm(self.W_q(x))).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        K = F.silu(self.k_norm(self.W_k(x))).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        V = self.W_v(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        beta = torch.sigmoid(self.W_beta(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)
        alpha = torch.sigmoid(self.W_alpha(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)
        scale = 1.0 / math.sqrt(d_h)
        
        # Track 1: Symmetrical Local Error-Feedback Gradient Pass
        G = torch.matmul(K, K.transpose(-1, -2)) * scale
        causal_mask = torch.tril(torch.ones(C, C, device=x.device), diagonal=-1).view(1, 1, 1, C, C)
        V_corrected = V - torch.matmul(G * causal_mask, V) * beta
        
        A_local = torch.matmul(Q, K.transpose(-1, -2)) * scale
        Y_local = torch.matmul(A_local * causal_mask, V_corrected)
        
        # Track 2: Cross-Chunk Global Error Memory Handoff
        log_alpha = torch.log(alpha.clamp(min=1e-6))
        decay_intra = torch.exp(torch.cumsum(log_alpha, dim=-2))
        
        Q_g, K_g, V_g = Q * decay_intra, K / (decay_intra + 1e-6), V_corrected * (1.0 - alpha)
        
        A_k = decay_intra[:, :, :, -1:, :]
        B_k = torch.matmul(K_g.transpose(-1, -2), V_g)
        
        # --- FIX: Explicitly lock the 5D tracking geometry ---
        A_k_expand = A_k.view(B, H, N, 1, 1)
        log_A_chunk = torch.log(A_k_expand.clamp(min=1e-6))
        A_chunk_cumsum = torch.exp(torch.cumsum(log_A_chunk, dim=2))
        
        # Clean parallel matrix transformations
        B_bar = A_k_expand * B_k
        B_scaled = B_bar / (A_chunk_cumsum + 1e-6)
        S_all = torch.cumsum(B_scaled, dim=2) * A_chunk_cumsum
        
        zeros = torch.zeros(B, H, 1, d_h, d_h, device=x.device, dtype=x.dtype)
        Y_global = torch.matmul(Q_g, torch.cat([zeros, S_all[:, :, :-1]], dim=2))
        
        Y_local_flat = Y_local.permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        Y_global_flat = Y_global.permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        
        Out = Y_local_flat + (Y_global_flat * torch.sigmoid(self.W_gate(x)))
        return self.W_out(Out)

    def step_inference(self, x, past_S=None):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        
        Q = F.silu(self.q_norm(self.W_q(x)).view(B, H, 1, d_h))
        K = F.silu(self.k_norm(self.W_k(x)).view(B, H, 1, d_h))
        V = self.W_v(x).view(B, H, 1, d_h)
        beta, alpha = torch.sigmoid(self.W_beta(x)).view(B, H, 1, 1), torch.sigmoid(self.W_alpha(x)).view(B, H, 1, 1)
        scale = 1.0 / math.sqrt(d_h)
        
        if past_S is None:
            S_global, K_cache, V_cache, step_idx = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype), [], [], 0
        else:
            S_global, K_cache, V_cache, step_idx = past_S
            
        K_cache.append(K); V_cache.append(V)
        K_hist, V_hist = torch.cat(K_cache, dim=2), torch.cat(V_cache, dim=2)
        
        V_corrected = V - (torch.matmul(torch.matmul(K, K_hist[:, :, :-1].transpose(-1, -2)) * scale, V_hist[:, :, :-1]) * beta) if step_idx % C > 0 else V
        
        Y_local = torch.matmul(torch.matmul(Q, K_hist.transpose(-1, -2)) * scale, V_hist)
        S_global = alpha * (S_global + torch.matmul(K.transpose(-1, -2), V_corrected * (1.0 - alpha)))
        
        Out = (Y_local.permute(0, 2, 1, 3).view(B, L, self.inner_dim) + 
               (torch.matmul(Q, S_global).permute(0, 2, 1, 3).view(B, L, self.inner_dim) * torch.sigmoid(self.W_gate(x))))
        
        step_idx += 1
        if step_idx % C == 0: K_cache, V_cache = [], []
        return self.W_out(Out), (S_global, K_cache, V_cache, step_idx)


class COFEDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalOrthogonalFeedbackEngine(d_model, num_heads, chunk_size=chunk_size)
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


class CausalCOFELM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            COFEDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        chunk_size = self.layers[0].attn.chunk_size
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        if pad_len > 0: x = F.pad(x, (0, pad_len), value=0)
            
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h))[:, :L, :]

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