import torch
from torch import nn
import torch.nn.functional as F

from .dilated_attention import DilatedAttention

class DilatedAttentionBlock(nn.Module):

    def __init__(self, w_r_sequence, hidden_size, num_attention_heads, dropout_p=0.0, scale=None):

        super().__init__()

        self.w_r_sequence = w_r_sequence

        self.dilated_attention_block = torch.nn.ModuleList()

        for (w, r) in self.w_r_sequence:
            self.dilated_attention_block.append(DilatedAttention(w, r, hidden_size, num_attention_heads, dropout_p=dropout_p, scale=scale))

    def forward(self, X, causal=False):

        output = torch.zeros(X.shape, device=X.device, dtype=X.dtype)
        # TODO this operation should be parallelized
        for i, l in enumerate(self.dilated_attention_block):
            output += l(X, causal=causal, normalize_with_softmax=False)

        return output
