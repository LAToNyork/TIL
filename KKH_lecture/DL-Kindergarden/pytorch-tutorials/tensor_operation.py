# PyTorch Tensor Operations
import torch

# Arithmetic Element-wise Operations
a = torch.FloatTensor([[1, 2],
                       [3, 4]])
b = torch.FloatTensor([[2, 2],
                       [3, 3]])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a == b)
print(a != b)
print(a ** b)

# Inplace Operations
print(a)
print(a.mul(b))
print(a)
print(a.mul_(b))
print(a)

# Sum, Mena (Dimnension Reducing Operations)
x = torch.FloatTensor([[1, 2], 
                       [3, 4]])
print(x.sum())
print(x.mean())

print(x.sum(dim=0))
print(x.mean(-1))

# Broadcast in Operations
x = torch.FloatTensor([[1, 2]])
y = torch.FloatTensor([[4, 8]])

print(x.size())
print(y.size())

z = x + y
print(z)
print(z.size())

# Tensor + Scalar
x = torch.FloatTensor([[1, 2],
                       [3, 4]])

y = 1
print(x.size())

z = x + y
print(z)
print(z.size())

# Tensor + Vector
x = torch.FloatTensor([[1, 2], 
                       [4, 8]])
y = torch.FloatTensor([3,
                       5])
print(x.size())
print(y.size())

z = x + y
print(z)
print(z.size())

x = torch.FloatTensor([[[1, 2]]])
y = torch.FloatTensor([3,
                       5])
print(x.size())
print(y.size())

z = x + y
print(z)
print(z.size())

# Tensor + Tensor
x = torch.FloatTensor([[1, 2]])
y = torch.FloatTensor([[3],
                       [5]])

print(x.size())
print(y.size())

z = x + y
print(z)
print(z.size())

# Failure case
x = torch.FloatTensor([[[1, 2],
                        [4, 8]]])
y = torch.FloatTensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

print(x.size())
print(y.size())

# z = x + y