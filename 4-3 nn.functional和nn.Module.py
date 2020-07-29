from torchkeras import summary
import datetime
import torch
import torch.nn.functional as F
from torch import nn

# 打印时间


def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % nowtime)

# 使用nn.Module来管理参数


w = nn.Parameter(torch.randn(2, 2))
# 自动具有求导属性

params_list = nn.ParameterList(
    [nn.Parameter(torch.rand(8, i)) for i in range(1, 3)])
# 将多个参数组成列表

params_dict = nn.ParameterDict({'a': nn.Parameter(torch.rand(8, 2)),
                                'b': nn.Parameter(torch.zeros(2))})
# 将多个参数组成字典

module = nn.Module()
module.w = w
module.params_list = params_list
module.params_dict = params_dict
for param in module.parameters():
    print(param)
# 使用module来统一管理参数


# nn.Linear源码的简化版实现
class Linear(nn.Module):
    __constant__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
# 可以看到，pt的做法是将层的参数与该层绑定在一起，创建参数时，使用nn.Parameter，此方法默认使该参数可求导

# 在实践中，很少直接使用nn.Parameter来定义模型，一般都是直接使用层来构建，这些层也是继承nn.Module的对象
# 提供了一些方法返回子模块的生成器


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=10000, embedding_dim=3, padding_idx=1)
        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=5))
        self.conv.add_module('pool1', nn.MaxPool1d(kernel_size=2))
        self.conv.add_module('relu1', nn.ReLU())
        self.conv.add_module('conv2', nn.Conv1d(
            in_channels=16, out_channels=128, kernel_size=2))
        self.conv.add_module('pool2', nn.MaxPool1d(kernel_size=2))
        self.conv.add_module('relu2', nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module('flatten', nn.Flatten())
        self.dense.add_module('linear', nn.Linear(6144, 1))
        self.dense.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y


net = Net()

i = 0
for child in net.children():  # 返回模块下的所有子模块
    i += 1
    print(child, '\n')
print('child number:', i)

i = 0
for name, child in net.named_children():  # 返回模块下所有子模块以及名字
    i += 1
    print(name, ':', child, '\n')
print('child number:', i)

summary(net, input_shape=(200,), input_dtype=torch.LongTensor)
