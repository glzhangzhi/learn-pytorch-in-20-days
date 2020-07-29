# PyTorch官方那个并没有更加高阶的API，训练验证和预测都需要用户自己实现
# 所以可以自定定义一些高阶API
# 分别实现fit, validate, predict, summary

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torchkeras
from torchkeras import Model
import torch.nn.functional as F
from torch import nn
import torch
import datetime
from torchkeras import Model


def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % nowtime)

# 1，准备数据


# 样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n, 2])-5.0  # torch.rand是均匀分布
w0 = torch.tensor([[2.0], [-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal(0.0, 2.0, size=[n, 1])  # @表示矩阵乘法,增加正态扰动

# 构建输入数据管道
ds = TensorDataset(X, Y)
ds_train, ds_valid = torch.utils.data.random_split(
    ds, [int(400*0.7), 400-int(400*0.7)])
dl_train = DataLoader(ds_train, batch_size=10, shuffle=True, num_workers=2)
dl_valid = DataLoader(ds_valid, batch_size=10, num_workers=2)

# 继承用户自定义模型


class LinearRegression(Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


model = LinearRegression()

model.summary(input_shape=(2,))

# 使用fit方法进行训练


def mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred-y_true))


def mean_absolute_percent_error(y_pred, y_true):
    absolute_percent_error = (
        torch.abs(y_pred-y_true)+1e-7)/(torch.abs(y_true)+1e-7)
    return torch.mean(absolute_percent_error)


model.compile(loss_func=nn.MSELoss(),
              optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
              metrics_dict={"mae": mean_absolute_error, "mape": mean_absolute_percent_error})

dfhistory = model.fit(200, dl_train=dl_train,
                      dl_val=dl_valid, log_step_freq=20)

# 四。评估模型

dfhistory.tail()

model.evaluate(dl_valid)

# 5，使用模型

# 预测
dl = DataLoader(TensorDataset(X))
model.predict(dl)[0:10]

# 预测
model.predict(dl_valid)[0:10]

# 二，DNN二分类模型
# 此范例我们通过继承上述用户自定义 Model模型接口，实现DNN二分类模型。

# 1，准备数据

# 正负样本数量
n_positive, n_negative = 2000, 2000

# 生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0, 1.0, size=[n_positive, 1])
theta_p = 2*np.pi*torch.rand([n_positive, 1])
Xp = torch.cat([r_p*torch.cos(theta_p), r_p*torch.sin(theta_p)], axis=1)
Yp = torch.ones_like(r_p)

# 生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0, 1.0, size=[n_negative, 1])
theta_n = 2*np.pi*torch.rand([n_negative, 1])
Xn = torch.cat([r_n*torch.cos(theta_n), r_n*torch.sin(theta_n)], axis=1)
Yn = torch.zeros_like(r_n)

# 汇总样本
X = torch.cat([Xp, Xn], axis=0)
Y = torch.cat([Yp, Yn], axis=0)

ds = TensorDataset(X, Y)

ds_train, ds_valid = torch.utils.data.random_split(
    ds, [int(len(ds)*0.7), len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=2)
dl_valid = DataLoader(ds_valid, batch_size=100, num_workers=2)

# 2，定义模型


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y


model = torchkeras.Model(Net())
model.summary(input_shape=(2,))

# 3，训练模型

# 准确率


def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                         torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc


model.compile(loss_func=nn.BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
              metrics_dict={"accuracy": accuracy})

dfhistory = model.fit(100, dl_train=dl_train,
                      dl_val=dl_valid, log_step_freq=10)

# 四。评估模型

model.evaluate(dl_valid)

# 5，使用模型

model.predict(dl_valid)[0:10]
