import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.base_model import RMSNorm, GatedMLP

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) natively in PyTorch.
    Rotates Query and Key feature pairs based on relative token distance.
    """
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, q, k, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(1) # [B, 1, L, d_h]
        sin = self.sin_cached[position_ids].unsqueeze(1) # [B, 1, L, d_h]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    SOTA Grouped-Query Attention (GQA) with fused Rotary Position Embeddings (RoPE).
    Utilizes PyTorch's native hardware-accelerated Scaled Dot-Product Attention (SDPA).
    """
    def __init__(self, d_model, num_heads=16, num_kv_heads=4, dropout=0.1, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        assert num_heads % num_kv_heads == 0, "Query heads must be perfectly divisible by KV heads."
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len)
        
        self.dropout_p = dropout if self.training else 0.0
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, position_ids=None):
        B, L, D = x.shape
        H_q, H_kv, d_h = self.num_heads, self.num_kv_heads, self.head_dim
        
        if position_ids == None:
            past_len = past_kv[0].size(-2) if past_kv is not None else 0
            position_ids = torch.arange(past_len, past_len + L, dtype=torch.long, device=x.device).unsqueeze(0)
            
        q = self.W_q(x).view(B, L, H_q, d_h).transpose(1, 2)
        k = self.W_k(x).view(B, L, H_kv, d_h).transpose(1, 2)
        v = self.W_v(x).view(B, L, H_kv, d_h).transpose(1, 2)
        
        q, k = self.rope.apply_rope(q, k, position_ids)
        
        if past_kv is not None:
            pk, pv = past_kv
            k, v = torch.cat([pk, k], dim=-2), torch.cat([pv, v], dim=-2)
        current_kv = (k, v)
        
        if H_kv != H_q:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1)
            
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=(past_kv is None)
        )
        
        out_flat = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_drop(self.W_out(out_flat)), current_kv


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=4, dropout=0.1, max_seq_len=4096, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, num_kv_heads, dropout, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x, past_kv=None, position_ids=None):
        def _block_forward(x, past_kv, position_ids):
            attn_out, current_kv = self.attn(self.ln1(x), past_kv=past_kv, position_ids=position_ids)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, current_kv
        
        if self.use_grad_checkpointing:
            return checkpoint.checkpoint(_block_forward, x, past_kv, position_ids, use_reentrant=False)
        else:
            return _block_forward(x, past_kv, position_ids)


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, max_seq_len=4096, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        num_kv_heads = 4 if num_heads >= 4 else 1
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, num_kv_heads, dropout=dropout, max_seq_len=max_seq_len, use_grad_checkpointing=use_grad_checkpointing) 
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        position_ids = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
        h = self.drop(self.token_embedding(x))
        for layer in self.layers:
            h, _ = layer(h, position_ids=position_ids)
        return self.lm_head(self.final_norm(h))

    def step_inference(self, x, past_states=None):
        h = self.token_embedding(x)
        next_states = []
        if past_states is None: past_states = [None] * len(self.layers)
            
        for i, block in enumerate(self.layers):
            past_kv = past_states[i]
            past_len = past_kv[0].size(-2) if past_kv is not None else 0
            position_ids = torch.tensor([[past_len]], dtype=torch.long, device=x.device)
            
            h_attn = block.ln1(h)
            attn_out, current_kv = block.attn(h_attn, past_kv=past_kv, position_ids=position_ids)
            h = h + attn_out
            h = h + block.mlp(block.ln2(h))
            next_states.append(current_kv)
            
        return self.lm_head(self.final_norm(h)), next_states