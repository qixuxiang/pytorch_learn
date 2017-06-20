# -*- coding:utf-8 -*-
import torch#基本的torch函数
import torch.autograd as autograd#自动求导

x = autograd.Variable(torch.ones(2, 2), requires_grad=True)
y = x+2
z = pow(y,3)
ave = z.mean()
ave.backward()
print(x.grad)
'''
Variable containing:
 6.7500  6.7500
 6.7500  6.7500
[torch.FloatTensor of size 2x2]
'''
x11 = torch.randn(3)
x11 = autograd.Variable(x11, requires_grad=True)
print(x11)
y11 = x11 * 2
while y11.data.norm() < 1000:
    y11 = y11 * 2

print('y11 is',y11)