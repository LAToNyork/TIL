# Activation functions 
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

x = torch.sort(torch.randn(100) * 10)[0]

print(x)

# Sigmoid
act = nn.Sigmoid()
print(act(x))

print(torch.sigmoid(x))

plt.plot(x.numpy(), torch.sigmoid(x).numpy())
plt.show()

# Hyperbolic Tangent (TanH)
act = nn.Tanh()
print(act(x))

print(torch.tanh(x))

plt.plot(x.numpy(), torch.tanh(x).numpy())
plt.show()