# PyTorch Tensor Manipulations
import torch

# Tensor shaping (reshape: change tensor shape)
x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                       [11, 12]]])
print(x.size())

print(x.reshape(12)) # 12 = 3 * 3 * 2
print(x.reshape(-1))

print(x.reshape(3, 4)) # 3 * 4 = 3 * 2 * 2
print(x.reshape(3, -1))

print(x.reshape(3, 1, 4))
print(x.reshape(-1, 1, 4))

print(x.reshape(3, 2, 2, 1))

# view = reshape

# squeeze
x = torch.FloatTensor([[[1, 2],
                        [3, 4]]])
print(x.size())

print(x.squeeze(0).size())
print(x.squeeze(1).size())

# unsqueeze
x = torch.FloatTensor([[1, 2],
                       [3, 4]])
print(x.size())

print(x.unsqueeze(2).size())
print(x.unsqueeze(-1).size())
print(x.reshape(2, 2, -1))