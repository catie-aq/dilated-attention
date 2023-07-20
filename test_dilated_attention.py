import torch

from longnet import DilatedAttention, DilatedAttentionBlock

X = torch.rand((5, 16384*32, 768)).to(torch.bfloat16).cuda()

dab = DilatedAttentionBlock([(128, 128), (256,128), (512,128), (1024, 128)], hidden_size=768, num_attention_heads=12).to(torch.bfloat16).cuda()
da = DilatedAttention(128, 128, hidden_size=768, num_attention_heads=12).to(torch.bfloat16).cuda()

print(dab(X).shape)
