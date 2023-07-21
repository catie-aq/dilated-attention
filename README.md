## Dilated_Attention - Pytorch (WIP)

This an unofficial implementation (WIP) of the dilated attention mechanism described in the <a href="https://arxiv.org/abs/2307.02486">LongNet</a> paper.
It rely on <a href="https://github.com/Dao-AILab/flash-attention">FlashAttention 2</a> for the dense attention implementation.

## Install

```bash
$ pip install dilated-attention
```

## Usage

```python
import torch

from dilated_attention import DilatedAttention, DilatedAttentionBlock

X = torch.rand((5, 16384, 768)).to(torch.bfloat16).cuda() # the implementation of FlashAttention supports only fp16 or bf16

da = DilatedAttention(128, # size of the segment (should divide the sequence length)
                      16, # dilation
                    hidden_size=768, # hidden_size dimension
                    num_attention_heads=12 # number of attention heads
                    ).to(torch.bfloat16).cuda()

dab(X) # (5, 16384, 768)

```

## TODO

- [ ] BUG : attention_probs returned by FlashAttention is None preventing to normalize the attention as described in the paper.
- [ ] check the implementation
- [ ] optimise the implementation (especially the sparsification ?)
- [ ] parallelize the DilatedAttentionBlock
- [ ] parallelization by splitting the input sequence on multiple GPUs as decribed in the paper
- [ ] supports for variable length using an attention mask ?
- [ ] benchmark the code and compare with the results of the paper

## Citations

```bibtex
@article{longnet,
    title   = {LongNet: Scaling Transformers to 1,000,000,000 Tokens},
    author  = {Jiayu Ding and Shuming Ma and Li Dong and Xingxing Zhang and Shaohan Huang, Wenhui Wang and Nanning Zheng and Furu Wei},
    year    = {2023},
    eprint  = {2307.02486},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning,
  author={Dao, Tri},
  year={2023},
  url  = {https://tridao.me/publications/flash2/flash2.pdf}
}
```
