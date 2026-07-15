import torch
import torch.nn as nn
from fla.layers import Mamba3
from models.base_model import RMSNorm, GatedMLP

class Mamba3DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Mamba3(
            hidden_size=d_model,
            expand=2,
            head_dim=64
        )
        self.ln2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_ff=int(d_model * 3.5), dropout=dropout)

    def forward(self, x):
        # We perform the forward pass directly here. 
        # Checkpointing will be handled at the model level only if safe.
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class CausalMamba3LM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers=8, dropout=0.1, use_grad_checkpointing=False):
        super().__init__()
        # We explicitly acknowledge that Mamba3 kernels do not support re-entrant or standard checkpointing
        self.use_grad_checkpointing = False 
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Mamba3DecoderBlock(d_model, num_heads=num_heads, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h))

    def step_inference(self, x, past_states=None):
        h = self.token_embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.final_norm(h)), past_states