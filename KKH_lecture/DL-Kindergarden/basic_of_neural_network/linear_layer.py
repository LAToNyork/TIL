import torch

# Raw Liner Layer
W = torch.FloatTensor([[1, 2],
                       [3, 4],
                       [5, 6]])
b = torch.FloatTensor([2, 2])

print(W.size())
print(b.size())

def linear(x, W, b):

    y = torch.matmul(x, W) + b

    return y

X = torch.FloatTensor([[1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3],
                       [4, 4, 4]])

print(X.size())

y = linear(X, W, b)
print(y.size())

# nn.module
import torch.nn as nn

class MyLinear(nn.Module):

    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.W = torch.FloatTensor(input_dim, output_dim)
        self.b = torch.FloatTensor(output_dim)

    def forward(self, X):
        # |X| = (batch_size, input_dim)
        y = torch.matmul(X, self.W) + self.b
        # |y| = (batch_size, input_dim) * (input_dim, output_dim)
        #     = (batch_size, output_dim)

        return y

linear = MyLinear(3, 2)

y = linear(X)

print(y.size())

for p in linear.parameters():
    print(p)

# Correct nn.Module
class MyLinear(nn.Module):

    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.b = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, X):
        # |X| = (batch_size, input_dim)
        y = torch.matmul(X, self.W) + self.b
        # |y| = (batch_size, input_dim) * (input_dim, output_dim)
        #     = (batch_size, output_dim)

        return y

linear = MyLinear(3, 2)

y = linear(X)

print(y.size())

for p in linear.parameters():
    print(p)

# nn.Linear
linear = nn.Linear(3, 2)

y = linear(X)

print(y.size())

for p in linear.parameters():
    print(p)

# nn.Module can contain other nn.Module's child classes
class MyLinear(nn.Module):

    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        # |X| = (batch_size, input_dim)
        y = self.linear(X)
        # |y| = (batch_size, input_dim) * (input_dim, output_dim)
        #     = (batch_size, output_dim)

        return y

linear = MyLinear(3, 2)

y = linear(X)

print(y.size())

for p in linear.parameters():
    print(p)