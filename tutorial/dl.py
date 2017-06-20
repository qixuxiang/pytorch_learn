# -*- coding:utf-8 -*-
import torch#基本的torch函数
import torch.autograd as autograd#自动求导
import torch.nn as nn#神经网络类都在这个里面
import torch.nn.functional as F#几乎所有的激励函数
import torch.optim as optim#优化

'''
Affine maps
也可以说是线性映射，即为f(x) = Ax + b
nn.Linear(inputSize,outputSize,bias=True)
输入(N, inputSize)
输出(N, outputSize)
'''
lin = nn.Linear(5,3)
print(lin)
data = autograd.Variable(torch.randn(2, 5))
print('lin(data) is',lin(data))

'''
Non-Linearities
非线性，常用的函数有 tanh(x),σ(x),ReLU(x) 这些都是激励函数
在pytorch中大部分激励函数在torch.functional中
'''
data = autograd.Variable(torch.randn(2, 2))
print('data is',data)
print (F.relu(data))#relu函数是小于零是0，大于零就是它本身


'''
Softmax and Probabilities
softmax是x_i/sum(x)
'''
data = autograd.Variable(torch.randn(5))
print('data is',data)
print(F.softmax(data))#归一化
print(F.softmax(data).sum())#和为1
print(F.log_softmax(data))
