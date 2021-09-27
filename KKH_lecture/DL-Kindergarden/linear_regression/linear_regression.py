# Linear Regression
# Load Dataset from sklearn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()

print(boston.DESCR)

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target
print(df.tail())

sns.pairplot(df)
#plt.show()

cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]
df[cols].describe

sns.pairplot(df[cols])
#plt.show()

# Train Linear Model with PyTorch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df[cols].values).float()

print(data.shape)

# split X and y
y = data[:, :1]
x = data[:, 1:]

print(x.shape, y.shape)

# Define configurations
n_epochs = 1000
learning_rate = 1e-3
print_interval = 100

# Define model
model = nn.Linear(x.size(-1), y.size(-1))

print(model)

# Instead of Implement gradient equation
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Whole training samples are used in 1 epochs
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i + 1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' % (i + 1, loss))

# Check the result
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])

sns.pairplot(df,height=5)
plt.show()