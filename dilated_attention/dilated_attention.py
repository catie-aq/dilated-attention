import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math

from flash_attn import flash_attn_kvpacked_func

class DilatedAttention(nn.Module):

    def __init__(self, segment_length, dilation, hidden_size, num_attention_heads, dropout_p=0.0, scale=None):

        super().__init__()

        self.w = segment_length
        self.r = dilation
        self.num_points = int(segment_length / dilation)

        assert hidden_size % num_attention_heads == 0, "Hidden size if not divisible in " + str(num_attention_heads) + " heads"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.dropout_p = dropout_p

        if scale is None:
            self.scale = 1 / math.sqrt(self.attention_head_size)
        else:
            self.scale = scale

        self.kv_layer = nn.Linear(hidden_size, 2*hidden_size)
        self.q_layer = nn.Linear(hidden_size, hidden_size)


    def forward(self, X, causal=False, normalize_with_softmax=False):
        """
            X: (batch_size, seqlen, hidden_size)
            causal : bool - ausal mask passed to the
        """
        X = X.contiguous()

        batch_size, seqlen = X.shape[:2]
        assert seqlen % self.w == 0, "The sequence length is not divisible by the segment length."
        num_segments = int(seqlen / self.w)

        # compute key queries and values
        kv = self.kv_layer(X).reshape((batch_size, seqlen, 2, self.num_attention_heads, self.attention_head_size))
        q = self.q_layer(X).reshape((batch_size, seqlen, self.num_attention_heads, self.attention_head_size))

        indices = [torch.linspace(k*self.w, (k+1)*self.w - self.r, self.num_points, dtype=torch.long).to(X.device) for k in range(0, num_segments)]

        index_tensor = torch.stack(indices, dim=0) # [num_segments, r]
        index_tensor = index_tensor.flatten() # [num_segments * r]
        shift_heads = torch.Tensor([i % self.r for i in range(self.num_attention_heads)]).long().to(X.device) # shift head i indices by i % r

        indices_with_heads = index_tensor.repeat(self.num_attention_heads, 1) + shift_heads.unsqueeze(-1) # [num_heads, num_segments * r]

        indices_q = indices_with_heads.expand(batch_size, self.attention_head_size, -1, -1).permute(0, 3, 2 , 1) # [batch_size, num_segments * r, num_heads, attention_head_size]
        indices_kv = indices_with_heads.expand(batch_size, self.attention_head_size, 2, -1, -1).permute(0, 4, 2 , 3, 1) # [batch_size, num_segments * r, 2, num_heads, attention_head_size]

        sparse_kv = torch.gather(kv, 1, indices_kv)
        sparse_q = torch.gather(q, 1, indices_q)

        # gather all key value for distributed - not implemented
        #if dist.is_initialized():
        #    world_size = dist.get_world_size()
        #    if world_size > 1:
        #        sparse_kv = torch.cat(gather(sparse_kv), dim=1)

        sparse_kv = sparse_kv.reshape((batch_size*num_segments, self.num_points,  2, self.num_attention_heads, self.attention_head_size)) # [batch_size * num_segments,  r, 2, num_heads, attention_head_size]
        sparse_q = sparse_q.reshape((batch_size*num_segments, self.num_points, self.num_attention_heads, self.attention_head_size)) # [batch_size * num_segments,  r, num_heads, attention_head_size]

        context, softmax_lse, attention_probs = flash_attn_kvpacked_func(sparse_q, sparse_kv, dropout_p=self.dropout_p, softmax_scale=self.scale, causal=causal, return_attn_probs=True) # [batch_size * num_segments,  r, num_heads, attention_head_size]
        context = context.reshape((batch_size, num_segments * self.num_points, self.num_attention_heads, self.attention_head_size)) # [batch_size, num_segments *  r, num_heads, attention_head_size]

        # normalize the new vectors with the softmax probability, allow to sum the vector in an attention_block
        if normalize_with_softmax:
            factor = 1.0 / attention_probs.reshape((batch_size, num_segments * self.num_points, self.num_attention_heads, self.attention_head_size))
            context = context * factor

        # scatter back to the original shape
        scatter_tensor = torch.zeros((batch_size, seqlen, self.num_attention_heads, self.attention_head_size), dtype=context.dtype, device=context.device)
        scatter_tensor = scatter_tensor.scatter(1, indices_q, context).reshape((batch_size, seqlen, self.hidden_size))

        return scatter_tensor
