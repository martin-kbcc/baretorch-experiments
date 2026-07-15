import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

class LowRankAssociativeDeltaEngine(nn.Module):
    def __init__(self, d_model=256, num_heads=16, chunk_size=32, rank=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.r = rank  
        
        self.d_head = d_model // num_heads
        self.inner_dim = self.num_heads * self.d_head
        
        self.W_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        self.W_u = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        self.W_r = nn.Linear(d_model, self.num_heads * self.r, bias=False)
        
        self.W_gate = nn.Linear(d_model, num_heads, bias=True)        
        self.W_beta_gate = nn.Linear(d_model, num_heads, bias=True)  
        
        self.W_swish_gate = nn.Linear(d_model, self.inner_dim, bias=False)
        self.W_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H, C, d_h, r = self.num_heads, self.chunk_size, self.d_head, self.r
        N = L // C  
        
        Q = F.silu(self.W_q(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        K = F.silu(self.W_k(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4))
        V = self.W_v(x).view(B, N, C, H, d_h).permute(0, 3, 1, 2, 4)
        
        U = self.W_u(x).view(B, N, C, H, r).permute(0, 3, 1, 2, 4)   
        R = self.W_r(x).view(B, N, C, H, r).permute(0, 3, 1, 2, 4)   
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1), min=1e-3, max=0.999)
        beta_gate = torch.sigmoid(self.W_beta_gate(x)).view(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)
        
        log_gate = torch.log(gate)
        Lambda = torch.cumsum(log_gate, dim=-2)
        exp_Lambda = torch.exp(Lambda)  
        
        causal_mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, 1, C, C)
        M_links = torch.exp((Lambda - Lambda.transpose(-1, -2)).masked_fill(causal_mask == 0, float('-inf')))
        
        scaling = 1.0 / math.sqrt(d_h)
        Y_local = torch.matmul(torch.matmul(Q, K.transpose(-1, -2)) * scaling * M_links, V)  
        
        chunk_decay_log = torch.sum(log_gate, dim=-2).squeeze(-1) 
        Lambda_chunks = torch.cumsum(chunk_decay_log, dim=2)  
        log_M_chunks = (Lambda_chunks.unsqueeze(-1) - Lambda_chunks.unsqueeze(-2)) - chunk_decay_log.unsqueeze(-1)
        
        causal_mask_chunks = torch.tril(torch.ones(N, N, device=x.device), diagonal=-1)
        M_chunks = torch.exp(log_M_chunks.masked_fill(causal_mask_chunks.view(1, 1, N, N) == 0, float('-inf')))
        
        U_decayed = (U * beta_gate) * (exp_Lambda[:, :, :, -1:, :] / torch.clamp(exp_Lambda, min=1e-6))
        S_historical = torch.matmul(M_chunks, torch.matmul(U_decayed.transpose(-1, -2), V).view(B, H, N, r * d_h)).view(B, H, N, r, d_h)
        
        Y_global = torch.matmul(R * exp_Lambda, S_historical) * scaling  
        
        Out = (Y_local + Y_global).permute(0, 2, 3, 1, 4).contiguous().view(B, L, self.inner_dim)
        return self.W_out(Out * F.silu(self.W_swish_gate(x)))

    def step_inference(self, x, past_S=None):
        B, L, D = x.shape
        H, d_h, r = self.num_heads, self.d_head, self.r
        
        Q = F.silu(self.W_q(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        K = F.silu(self.W_k(x).view(B, L, H, d_h).permute(0, 2, 1, 3))
        V = self.W_v(x).view(B, L, H, d_h).permute(0, 2, 1, 3)
        U, R = self.W_u(x).view(B, L, H, r).permute(0, 2, 1, 3), self.W_r(x).view(B, L, H, r).permute(0, 2, 1, 3)
        
        gate = torch.clamp(torch.sigmoid(self.W_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1), min=1e-3, max=0.999)
        beta_gate = torch.sigmoid(self.W_beta_gate(x)).view(B, L, H).permute(0, 2, 1).unsqueeze(-1)
        
        S_state = gate * (past_S if past_S is not None else torch.zeros(B, H, r, d_h, device=x.device, dtype=x.dtype)) + \
                  torch.matmul((U * beta_gate).transpose(-1, -2), V)
        
        Out = (torch.matmul(Q, torch.matmul(K.transpose(-1, -2), V)) + torch.matmul(R, S_state)) * (1.0 / math.sqrt(d_h))
        return self.W_out(Out.permute(0, 2, 1, 3).contiguous().view(B, L, D) * F.silu(self.W_swish_gate(x))), S_state


class LRADDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, chunk_size=32, rank=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = LowRankAssociativeDeltaEngine(d_model, num_heads, chunk_size=chunk_size, rank=rank)
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


class CausalLRADLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, chunk_size=32, rank=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LRADDecoderBlock(d_model, num_heads, chunk_size=chunk_size, rank=rank, dropout=dropout, use_grad_checkpointing=use_grad_checkpointing)
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