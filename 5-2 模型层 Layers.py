import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# 一般构造模型的砖块，即层，由两种方式提供：
# 一是内置的层，二是自定义层
# 自定义层只需要基层nn.Module类并覆写forward方法即可。


# 常用层
# nn.Linear
# nn.Flatten
# nn.BatchNorm2d
# nn.Dropout
# nn.Conv2d
# nn.MaxPool2d

# 自定义模型层


class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.laiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias in not None}'


linear = nn.Linear(20, 30)
inputs = torch.randn(128, 20)
outputs = linear(inputs)
print(outputs.size())
