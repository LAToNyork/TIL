# Matrix Multiplication
import torch

x = torch.FloatTensor([[1, 2],
                       [3, 4],
                       [5, 6]])
y = torch.FloatTensor([[1, 2],
                       [1, 2]])

print(x.size(), y.size())

z = torch.matmul(x, y)
print(x.size())

x = torch.FloatTensor([[[1, 2],
                        [3, 4],
                        [5, 6]],
                       [[7, 8],
                        [9, 10],
                        [11, 12]],
                       [[13, 14],
                        [15, 16],
                        [17, 18]]])
y = torch.FloatTensor([[[1, 2, 2],
                        [1, 3, 2]],
                       [[1, 3, 3],
                        [1, 3, 3]],
                       [[1, 4, 4],
                        [1, 4, 4]]])

print(x.size(), y.size())

z = torch.bmm(x, y)
print(z.size())