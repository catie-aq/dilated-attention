import torch

# set up some parameters
w = 32 # segment length
r = 10 # dilation


# Create a tensor of indices used to debug
batch_size = 5
seqlen = 320
num_heads = 12
head_size = 16

x = torch.empty(batch_size, seqlen, num_heads, head_size)
for i in range(batch_size):
  for j in range(seqlen):
    for k in range(num_heads):
        for l in range(head_size):
            x[i,j,k,l] = j
x = x.long()

assert seqlen % w == 0, "The sequence length is not divisible by the segment length."
num_segments = int(seqlen / w)

indices = [torch.linspace(k*w, (k+1)*w - r, r, dtype=torch.long) for k in range(0, num_segments)]
index_tensor = torch.stack(indices, dim=0)
index_tensor = index_tensor.flatten()
shift_heads = torch.Tensor([i % r for i in range(num_heads)]).long()

indices_with_heads = index_tensor.repeat(num_heads, 1) + shift_heads.unsqueeze(-1)

final_indices = indices_with_heads.expand(batch_size, head_size, -1, -1).permute(0, 3, 2 ,1)

final_tensor = torch.gather(x, 1, final_indices)
grouped_tensor = final_tensor.reshape((batch_size, num_segments, r, num_heads, head_size))
print(grouped_tensor[0, 1, :, 0, 0])

value_scatter_tensor = grouped_tensor.reshape((batch_size, num_segments * r, num_heads, head_size))

scatter_tensor = torch.zeros((batch_size, seqlen, num_heads, head_size), dtype=grouped_tensor.dtype)
scatter_tensor.scatter(1, final_indices, value_scatter_tensor).reshape((batch_size, seqlen, -1))
print(scatter_tensor.shape)
