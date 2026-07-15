import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

class CausalKernelGatedTensorSifter(nn.Module):
    """
    Causal Kernel-Gated Tensor Sifter (CKTS).
    Fuses high-expressivity local chunkwise causal attention matrices with an 
    anisotropic, linear global State-Space recurrence channel track. 
    Completely loop-free via parallel chunkwise cumulative sums.
    """
    def __init__(self, d_model=256, num_heads=16, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        # Track 1: Local Attention Projection Projections
        self.q_local = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_local = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_local = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # Track 2: Global State-Space Linear Recurrence Channels
        self.q_global = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_global = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_global = nn.Linear(d_model, self.inner_dim, bias=False)
        
        # State Control Gate Parameters
        self.alpha_proj = nn.Linear(d_model, self.num_heads, bias=True)
        self.gate_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        
        self.ln_q = RMSNorm(self.inner_dim)
        self.ln_k = RMSNorm(self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        N = L // C
        
        # Track 1: Local Chunkwise Non-Linear Softmax Attention Pass
        Ql = self.ln_q(self.q_local(x)).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        Kl = self.ln_k(self.k_local(x)).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        Vl = self.v_local(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        scores = torch.matmul(Ql, Kl.transpose(-1, -2)) * (1.0 / math.sqrt(d_h))
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        Y_local = torch.matmul(attn_weights, Vl)
        
        # Track 2: Linear Chunk-Parallel Global State-Space Recurrence
        Qg = self.ln_q(self.q_global(x)).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        Kg = torch.tanh(self.ln_k(self.k_global(x))).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4) * 0.4
        Vg = self.v_global(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)
        
        log_alpha = torch.log(alpha.clamp(min=1e-6))
        decay_intra = torch.exp(torch.cumsum(log_alpha, dim=-2))
        Qg_scaled = Qg * decay_intra
        Kg_scaled = Kg / (decay_intra + 1e-6)
        Vg_gated = Vg * (1.0 - alpha)
        
        A_k = decay_intra[:, :, :, -1:, :]
        B_k = torch.einsum('bhncd,bhncg->bhndg', Kg_scaled, Vg_gated)
        
        A_k_expand = A_k.view(B, H, N, 1, 1)
        log_A_chunk = torch.log(A_k_expand.clamp(min=1e-6))
        A_chunk_cumsum = torch.exp(torch.cumsum(log_A_chunk, dim=2))
        
        B_bar = A_k_expand * B_k
        B_scaled = B_bar / (A_chunk_cumsum + 1e-6)
        S_all = torch.cumsum(B_scaled, dim=2) * A_chunk_cumsum
        
        zeros = torch.zeros(B, H, 1, d_h, d_h, device=x.device, dtype=x.dtype)
        global_states = torch.cat([zeros, S_all[:, :, :-1]], dim=2)
        
        Y_global = torch.einsum('bhncd,bhndg->bhncg', Qg_scaled, global_states)
        
        Y_local_flat = Y_local.permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        Y_global_flat = Y_global.permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        
        fusion_gate = torch.sigmoid(self.gate_proj(x))
        Out = Y_local_flat + (Y_global_flat * fusion_gate)
        return self.out_proj(Out)

    def step_inference(self, x, past_S=None):
        B, L, D = x.shape
        H, C, d_h = self.num_heads, self.chunk_size, self.d_head
        
        Ql = self.ln_q(self.q_local(x)).view(B, H, 1, d_h)
        Kl = self.ln_k(self.k_local(x)).view(B, H, 1, d_h)
        Vl = self.v_local(x).view(B, H, 1, d_h)
        Qg = self.ln_q(self.q_global(x)).view(B, H, 1, d_h)
        Kg = torch.tanh(self.ln_k(self.k_global(x))).view(B, H, 1, d_h) * 0.4
        Vg = self.v_global(x).view(B, H, 1, d_h)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, H, 1, 1)
        
        if past_S is None:
            S_global = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
            K_local_cache, V_local_cache, step_idx = [], [], 0
        else:
            S_global, K_local_cache, V_local_cache, step_idx = past_S
            
        K_local_cache.append(Kl); V_local_cache.append(Vl)
        K_history = torch.cat(K_local_cache, dim=2)
        V_history = torch.cat(V_local_cache, dim=2)
        
        scores = torch.matmul(Ql, K_history.transpose(-1, -2)) * (1.0 / math.sqrt(d_h))
        Y_local = torch.matmul(F.softmax(scores, dim=-1), V_history)
        
        innovation = torch.matmul(Kg.transpose(-1, -2), Vg * (1.0 - alpha))
        S_global = alpha * (S_global + innovation)
        Y_global = torch.matmul(Qg, S_global)
        
        Out = Y_local.permute(0, 2, 1, 3).contiguous().view(B, L, self.inner_dim) + \
              (Y_global.permute(0, 2, 1, 3).contiguous().view(B, L, self.inner_dim) * torch.sigmoid(self.gate_proj(x)))
              
        step_idx += 1
        if step_idx % C == 0: K_local_cache, V_local_cache = [], []
        return self.out_proj(Out), (S_global, K_local_cache, V_local_cache, step_idx)


class CKTSDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalKernelGatedTensorSifter(d_model, num_heads, chunk_size=chunk_size)
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


class CausalCKTSLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CKTSDecoderBlock(d_model, num_heads, chunk_size=chunk_size, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
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