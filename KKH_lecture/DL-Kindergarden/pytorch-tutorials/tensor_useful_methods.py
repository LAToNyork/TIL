# PyTorch Useful Methods
import torch
from torch._C import Size

# expand
x = torch.FloatTensor([[[1, 2]],
                       [[3, 4]]])
print(x.size())

y = x.expand(2, 3, 2)

print(y)
print(y.size())

# randperm
x = torch.randperm(10)

print(x)
print(x.size())

# argmax
x = torch.randperm(3**3).reshape(3, 3, -1)

print(x)
print(x.size())

y = x.argmax(dim=-1)

print(y)
print(y.size())

# topk
values, indices = torch.topk(x, k=1, dim=-1)

print(values.size())
print(indices.size())

print(values.squeeze(-1))
print(indices.squeeze(-1))

print(x.argmax(dim=-1) == indices.squeeze(-1))

_, indices = torch.topk(x, k=2, dim=-1)
print(indices.size())

print(x.argmax(dim=-1) == indices[:, :, 0])

# sort by using topk
target_dim = -1
values, indices = torch.topk(x,
                             k=x.size(target_dim),
                             largest=True)

print(values)

k = 1
values, indices = torch.sort(x, dim=-1, descending=True)
values, indices = values[:, :, :k], indices[:, :, :k]

print(values.squeeze(-1))
print(indices.squeeze(-1))

# masked_fill
x = torch.FloatTensor([i for i in range(3**2)]).reshape(3, -1)

print(x)
print(x.size())

mask = x > 4
print(mask)

y = x.masked_fill(mask, value=-1)
print(y)

# ones and zeros
print(torch.ones(2, 3))
print(torch.zeros(2, 3))

x = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6]])

print(x.size())

print(torch.ones_like(x))
print(torch.zeros_like(x))