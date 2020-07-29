# 动态计算图的含义
# 1.图的构建是动态的，所有的前向传播是立即执行的，即定义运算的时候图已经成型了
# 2.反向传播是一次性的，在进行完一次反向传播后，计算图就会被销毁


# 一。动态性
from torch import nn
from tensorboard import notebook
from torch.utils.tensorboard import SummaryWriter
import torch
w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y, 2))

print(loss.data)
print(Y_hat.data)

w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y, 2))

# 计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
loss.backward()  # loss.backward(retain_graph = True)

# loss.backward() #如果再次执行反向传播将报错

# 二。计算图中的Function
# 计算图中的Function同时包含正向计算逻辑和反向传播逻辑
# 可以通过继承torch.autograd.Function来创建这种支持反向传播的Function


class MyReLU(torch.autograd.Function):

    # 正向传播逻辑，可以用ctx存储一些值，供反向传播使用。
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    # 反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
Y = torch.tensor([[2.0, 3.0]])

relu = MyReLU.apply  # relu现在也可以具有正向传播和反向传播功能
Y_hat = relu(X@w.t() + b)
loss = torch.mean(torch.pow(Y_hat-Y, 2))

loss.backward()

print(w.grad)
print(b.grad)

# Y_hat的梯度函数即是我们自己所定义的 MyReLU.backward

print(Y_hat.grad_fn)

# 三。利用TensorBoard进行可视化


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Parameter(torch.randn(2, 1))
        self.b = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        y = x@self.w + self.b
        return y


net = Net()
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net, input_to_model=torch.rand(10, 2))
writer.close()
%load_ext tensorboard
# %tensorboard --logdir ./data/tensorboard
notebook.list()
# 在tensorboard中查看模型
notebook.start("--logdir ./data/tensorboard")
