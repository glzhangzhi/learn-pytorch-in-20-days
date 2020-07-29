# pytorch可以自动完成求梯度运算
# 一般通过反向传播backward方法实现求梯度运算，求得的梯度将存在对应自变量张量的grad属性下
# 除此之外，也能够调用torch.autograd.grad函数来实现求梯度运算

# 一。利用backward方法求导数
# backward方法通常在一个标量张量上调用，如果调用的张量非标量，则要传入一个和它形状相同的gradian参数张量
# 相当于用该参数张量与调用张量做点乘，得到一个标量，再反向传播

import numpy as np
import torch

x = torch.tensor(0.0, requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a * torch.pow(x, 2) + b * x + c

y.backward()
dy_dx = x.grad
print(dy_dx)

x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a * torch.pow(x, 2) + b * x + c

gradian = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

y.backward(gradian=gradian)
dy_dx = x.grad
print(dy_dx)
'''
# 与上面效果相同
y = torch.sum(y*gradient)
y.backward()
'''

# 二。利用autograd.grad方法求导

dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]  # create_graph设为True将允许创建更高阶的导数
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx,x)[0]  # 为什么要加一个[0]？？？

print(dy2_dx2.data)

x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
y1 = x1 * x2
y2 = x1 + x2
# 同时对多个自变量求导
(dy1_dx1, dy1_dx2) = torch.autograd.grad(outputs=y1, inputs=[x1, x2], retain_graph=True)
(dy12_dx1, dy12_dx2) = torch.autograd.grad(outputs=[y1, y2], inputs=[x1, x2])  # 多个因变量求导，即把多个梯度结果求和

# 三。利用自动微分和优化器求最小值

# f(x) = a*x**2 + b*x + c的最小值

x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)
# 将需要被优化的参数设为自变量，即求最小值

def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
   
    
print("y=",f(x).data,";","x=",x.data)