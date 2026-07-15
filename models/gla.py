import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from fla.layers import GatedLinearAttention
from models.base_model import RMSNorm, GatedMLP

class GLADecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing
        self.ln1 = RMSNorm(d_model)
        self.attn = GatedLinearAttention(hidden_size=d_model, num_heads=num_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x):
        def _block_forward(x):
            # Note: fla.layers.GatedLinearAttention returns a tuple, we take the first element
            x = x + self.attn(self.ln1(x))[0]
            x = x + self.mlp(self.ln2(x))
            return x
        
        if self.use_grad_checkpointing:
            return checkpoint.checkpoint(_block_forward, x, use_reentrant=False)
        else:
            return _block_forward(x)


class CausalGLALM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GLADecoderBlock(
                d_model, 
                num_heads, 
                dropout=dropout, 
                use_grad_checkpointing=use_grad_checkpointing
            ) 
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h))