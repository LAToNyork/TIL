# Gradient descent
import torch
import torch.nn.functional as F

target = torch.FloatTensor([[.1, .2, .3],
                            [.4, .5, .6,],
                            [.7, .8, .9]])

x = torch.rand_like(target)
# This means the final scalar will be differentiate by x
x.requires_grad = True
# you can get gradient of X, after differentiate
print(x)

loss = F.mse_loss(x, target)

print(loss)

threshold = 1e-5
learning_rate = 1
iter_cnt = 0

while loss > threshold:
    iter_cnt += 1

    loss.backward() # Calculate gradients

    x = x - learning_rate * x.grad

    # you don't need to aware this now
    x.detach_()
    x.requires_grad_(True)

    loss = F.mse_loss(x, target)

    print("%d-th Loss: %.4e" % (iter_cnt, loss))
    print(x)