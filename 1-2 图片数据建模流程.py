# 一。数据准备

 
import os
import datetime

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)


 
import torch 
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets 

   
# pytorch中构建图片数据管道通常有两种方法：
# 
# 一是使用torchvision中的datasets.ImageFolder来读取图片然后用DataLoader来并行加载
# 
# 二是通过继承torch.utils.data.Dataset实现用户自定义读取逻辑然后用DataLoader来并行加载
# 
# 第二种方法是读取用户自定义数据集的通用方法，既可以读取图片，也可以读取文本
# 
# 这里使用第一种方法

 
transform_train = transforms.Compose([transforms.ToTensor()])
transform_valid = transforms.Compose([transforms.ToTensor()])
# 定义图片转换方式

   
# 图片文件路径结构：
# 
# cifar2
#   train
#     0_airplane
#       0.jpg
#       1.jpg
#       2.jpg
#       ...
#       4999.jpg
#     1_automobile
#       0.jpg
#       1.jpg
#       2.jpg
#       ...
#       4999.jpg
#   test
#     0_airplane
#       0.jpg
#       1.jpg
#       2.jpg
#       ...
#       999.jpg
#     1_automobile
#       0.jpg
#       1.jpg
#       2.jpg
#       ...
#       999.jpg

 
ds_train = datasets.ImageFolder('./data/cifar2/train/', transform=transform_train, target_transform=lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder('./data/cifar2/test/', transform=transform_valid, target_transform=lambda t:torch.tensor([t]).folat())
# 使用ImageFolder载入文件夹中的图片

print(ds_train.class_to_idx)
# 文件夹的类别自动onehot编码显示


 
dl_train = DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=3)
dl_test = DataLoader(ds_test, batch_size=40, shuffle=True, num_workers=3)


 
from matplotlib import pyplot as plt 

# 显示前10张图片
plt.figure(figsize=(8,8)) 
for i in range(9):
    img,label = ds_train[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label.item())
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()


 
for x, y in dl_train:
    print(x.shape, y.shape)
    # pytorch载入数据批次的默认顺序是m, c, w, h

   
# 二。定义模型
# 
# 这次使用继承nn.Module基类构建自定义模型

 
# 测试AdaptiveMaxPool2d效果
pool = nn.AdaptiveMaxPool2d((1, 1))
t = torch.randn(10, 8, 32, 32)
pool(t).shape
# 无论输入的尺寸是多少，输入的尺寸总是定义时写入的(1, 1)


 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.sigmoid(self.linear2(self.relu(self.linear1(self.flatten(self.adaptive_pool(self.dropout(self.conv2(self.pool(self.conv1(x))))))))))
        return y
    
net = Net()
print(net)
 
from torchkeras import summary
summary(net, input_shape=(3, 32, 32))

   
# 三。训练模型
# 
# 这里使用函数形式训练循环

 
import pandas as pd
from sklearn.metrics import roc_auc_score

model = net
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = torch.nn.BCELoss()
model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
model.metric_name = 'auc'

def train_step(model, features, labels):
    model.train()
    model.optimizer.zero_grad()
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()

def valid_step(model, features, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels)
        metric = model.metric_func(predictions, labels)
    
    return loss.item(), metric.item()

features, labels = next(iter(dl_train))
print(train_step(model, features, labels))

def train_model(model, epochs, dl_train, dl_valid, log_step_freq):
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'metric_name', 'val_loss', 'val_' + metric_name])
    print('Start Training')
    printbar()

    for epoch in range(1, epochs + 1):
        
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):
            loss, metric = train_step(model, features, labels)
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") % (step, loss_sum/step, metric_sum/step))

        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, labels)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/step, val_metric_sum/step)
        dfhistory.loc[epoch-1] = info

        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") %info)

        printbar()

    print('Finished Training...')

    return dfhistory

dfhistory = train_model(model, epochs=20, dl_train, dl_valid, log_step_freq=50)

# 四。评估模型
# 利用训练时记录下来的dfhistory来绘图评估模型的表现

import matplotlib.pyplot as plt

def plot_metric(dfhistory,metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro--')
    plt.title('Training and Validation '+ metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(['train_'+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory, 'loss')

plot_metric(dfhistory, 'auc')

# 五。使用模型

def predict(model, dl):
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in dl])
    return(result.data)

# 预测概率
y_pred_probs = predict(model, dl_valid)
print(y_pred_probs)

# 预测类别
y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
print(y_pred)

# 六。保存模型
# 推荐使用保存参数的方式保存模型

print(model.state_dict().keys())

torch.save(model.state_dict(), './data/model_parameter.pkl')

net_clone = Net()
net_clone.load_state_dict(torch.load('./data/model_parameter.pkl'))

predict(net_clone, dl_valid)