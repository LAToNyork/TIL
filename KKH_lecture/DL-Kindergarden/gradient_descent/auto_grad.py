# PyTorch Autograd
import torch

x = torch.FloatTensor([[1, 2], 
                       [3, 4]]).requires_grad_(True)

x1 = x + 2
x2 = x - 2
x3 = x1 * x2
y = x3.sum()

print(x1)
print(x2)
print(x3)
print(y)

print(y.backward())

print(x.grad)

print(x)

# x3.numpy()

x3.detach_().numpy()
print(x3)