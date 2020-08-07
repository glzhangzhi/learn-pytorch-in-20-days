'''
继承nn.Module基类构建自定义模型 最常见
使用nn.Sequential按层顺序构建 最简单
继承nn.Module基类构建模型并辅助应用模型容器进行封装 最灵活
'''
import torch 
from torch import nn
from torchkeras import summary


# 1
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
print(net)

summary(net,input_shape= (3,32,32))

# 2
net = nn.Sequential()
net.add_module("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3))
net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("dropout",nn.Dropout2d(p = 0.1))
net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
net.add_module("flatten",nn.Flatten())
net.add_module("linear1",nn.Linear(64,32))
net.add_module("relu",nn.ReLU())
net.add_module("linear2",nn.Linear(32,1))
net.add_module("sigmoid",nn.Sigmoid())
print(net)
# ---
net = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
    nn.Dropout2d(p = 0.1),
    nn.AdaptiveMaxPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1),
    nn.Sigmoid()
)
print(net)
# ---
from collections import OrderedDict
net = nn.Sequential(OrderedDict(
          [("conv1",nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)),
            ("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2)),
            ("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)),
            ("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2)),
            ("dropout",nn.Dropout2d(p = 0.1)),
            ("adaptive_pool",nn.AdaptiveMaxPool2d((1,1))),
            ("flatten",nn.Flatten()),
            ("linear1",nn.Linear(64,32)),
            ("relu",nn.ReLU()),
            ("linear2",nn.Linear(32,1)),
            ("sigmoid",nn.Sigmoid())
          ])
        )
print(net)

# 3
# nn.Sequential
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1))
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv(x)
        y = self.dense(x)
        return y 
    
net = Net()
print(net)

# nn.ModuleList
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()]
        )
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
net = Net()
print(net)
# nn.ModuleDict
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layers_dict = nn.ModuleDict({"conv1":nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
               "pool": nn.MaxPool2d(kernel_size = 2,stride = 2),
               "conv2":nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
               "dropout": nn.Dropout2d(p = 0.1),
               "adaptive":nn.AdaptiveMaxPool2d((1,1)),
               "flatten": nn.Flatten(),
               "linear1": nn.Linear(64,32),
               "relu":nn.ReLU(),
               "linear2": nn.Linear(32,1),
               "sigmoid": nn.Sigmoid()
              })
    def forward(self,x):
        layers = ["conv1","pool","conv2","pool","dropout","adaptive",
                  "flatten","linear1","relu","linear2","sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x
net = Net()
print(net)