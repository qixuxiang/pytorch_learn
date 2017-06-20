# -*- coding:utf-8 -*-
import torch#基本的torch函数
import torch.autograd as autograd#自动求导


#create 1D vector
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)#我用的是pyCharm编辑器，输入torch给的提示没有Tensor函数，其实是有的
print(V)


#create 2D vector
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.Tensor(M_data)
print(M)

#create 3D vector
T_data = [[[1.,2.], [3.,4.]],
          [[5.,6.], [7.,8.]]]
T = torch.Tensor(T_data)
print(T)


#我就觉得这里比TensorFlow好用多了QAQ
print(V[0])
print(M[0])
print(T[0])


r = torch.randn((3,4,5))
print(r)

#tensor计算
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
print('----test cuda begin----')
#Tensors can be moved onto GPU using the .cuda function
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)

print('---test cuda end----')
z = x + y
print(z)

x1 = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
print(x1)
print(x1.data)#.data显示具体数据
y1 = autograd.Variable( torch.Tensor([4., 5., 6]), requires_grad=True )
z1 = x1 + y1
print(z1.data)
s = z1.sum()
s.backward()#反向传播
print('x1.grad is:',x1.grad)#对x求导
'''
#答案解释
#x = [1,2,3]
#y = [4,5,6]
#z = x + y = [x0+y0, x1+y1, x2+y2]
#s = z.sum() = x0+y0+x1+y1+x2+y2
#x.grad 在s运算中对x求导   也就是当中的x0,x1,x2求导  为1，1，1
'''

x_1 = torch.randn(2, 5)#[torch.FloatTensor of size 2x5]
y_1 = torch.randn(3, 5)#[torch.FloatTensor of size 3x5]
z_1 =torch.cat([x_1, y_1])#没有最后一个参数，默认是0，则最终维度的第0维度为x_1与y_1第0维度的和，最终维度的其他维度不变.以下同理
print(x_1)
print(y_1)
print(z_1) #[torch.FloatTensor of size 5x5]

print(x_1.view(5,2)) # 5*2 -> 2*5[torch.FloatTensor of size 5x2]