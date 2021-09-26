# Mean Square Error (MSE) Loss
import torch

def mse(x_hat, x):
    # |x_hat| = (batch_size, dim)
    # |x| = (batch_size, dim)
    y  = ((x - x_hat)**2).mean()

    return y

x = torch.FloatTensor([[1, 1],
                       [2, 2]])
x_hat = torch.FloatTensor([[0, 0],
                           [0, 0]])

print(x.size(), x_hat.size())

print(mse(x_hat, x))

# Predifined MSE in PyTorch

# 함수
import torch.nn.functional as F

print(F.mse_loss(x_hat, x))

print(F.mse_loss(x_hat, x, reduction='sum'))

print(F.mse_loss(x_hat, x, reduction='none'))

# 객체
import torch.nn as nn

mse_loss = nn.MSELoss()

mse_loss(x_hat, x)

print(mse_loss(x_hat, x))