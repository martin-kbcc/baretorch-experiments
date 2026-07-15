import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
        return self.drop(out)


def sequence_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    tensor_ids = torch.tensor(input_ids, dtype=torch.long)
    x = tensor_ids[:, :-1]  
    y = tensor_ids[:, 1:]   
    return x, y