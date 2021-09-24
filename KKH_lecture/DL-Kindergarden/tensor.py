# PyTorch Tensor
import torch

# Tensor Allocation
ft = torch.FloatTensor([[1, 2],
                        [3, 4]])
print(ft)

# Integer Tensor
lt = torch.LongTensor([[1, 2],
                       [3, 4]])
print(lt)

# Boolean Tensor
bt = torch.ByteTensor([[1, 0],
                       [0, 1]])
print(bt)

x = torch.FloatTensor(3, 3)
print(x)

# NumPy Compatibility
import numpy as np

# define numpy array
x = np.array([[1, 2],
              [3, 4]])
print(x, type(x))

# ndarray -> Tensor
x = torch.from_numpy(x)
print(x, type(x))

# Tensor -> ndarray
x = x.numpy()
print(x, type(x))

# Tensor Type-casting

# ft -> long
ft.long()

# long -> ft
lt.float()

# byte
torch.FloatTensor([1, 0]).byte()

# Get shape
x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])
print(x.size())
print(x.shape)

print(x.dim())
print(len(x.shape))

print(x.size(1))
print(x.shape[1])

print(x.size(-1))
print(x.shape[-1])