# PyTorch Tensor Slicing and Concatenation
import torch

# Slicing and Concatenation

# Indexing and Slicing
x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])
print(x.size())

# Access to certain dimension
print(x[0])
print(x[0, :])
print(x[0, :, :])

print(x[-1])
print(x[-1, :])
print(x[-1, :, :])

print(x[:, 0, :])

# Access by range
print(x[1:3, :, :].size())
print(x[:, :1, :].size())
print(x[:, :-1, :].size())

# split
x = torch.FloatTensor(10, 4)

splits = x.split(4, dim=0)

for s in splits:
    print(s.size())

# chunk
x = torch.FloatTensor(8, 4)

chunks = x.chunk(3, dim=0)

for c in chunks:
    print(c.size())

# Index select
x = torch.FloatTensor([[[1, 1],
                        [2, 2]],
                       [[3, 3],
                        [4, 4]],
                       [[5, 5],
                        [6, 6]]])
indice = torch.LongTensor([2, 1])
print(x.size())

y = x.index_select(dim=0, index=indice)
print(y)
print(y.size())

# cat
x = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
y = torch.FloatTensor([[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]])

print(x.size(), y.size())

z = torch.cat([x, y], dim = 0)
print(z)
print(z.size())

z = torch.cat([x, y], dim = -1)
print(z)
print(z.size())

# stack
z = torch.stack([x, y])
print(z)
print(z.size())

z = torch.stack([x, y], dim=-1)
print(z)
print(z.size())

# Implement stack function by using cat
z = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
print(z)
print(z.size())

# useful trick
result = []
for i in range(5):
    x = torch.FloatTensor(2, 2)
    result += [x]

result = torch.stack(result)
result.size()