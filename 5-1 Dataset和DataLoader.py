# Dataset定义了数据集的内容，类似列表，具有确定的长度，使用索引获取元素
# 如果自己定义，只需要实现Dataset的__len__方法和__getitem__方法

# 获取一个batch的步骤及实现

# 1.确定数据集长度n
# 通过Dataset对象的__len__方法实现

# 2.从数据集中抽取m个数据的下标
# 默认设置

# 3.通过这m个下标获取对应的元素
# 通过Dataset对象的__getitem__方法实现

# 4.将结果整理成两个张量输出batch=(features, labels)
# 默认设置

# 核心逻辑
from torchvision import transforms, datasets
# from sklearn import datasets
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import torch


class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class DataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.sampler = torch.utils.data.RandomSampler if shuffle else \
            torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(range(len(dataset))),
            batch_size=batch_size, drop_last=drop_last
        )

    def __next__(self):
        indices = next(self.sample_iter)
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch


# 使用Dataset创建数据集的几种方法
# 1.使用Tensor创建

iris = datasets.load_iris()
ds_iris = TensorDataset(torch.tensor(iris.data), torch.tensor(iris.target))

n_train = int(len(ds_iris) * 0.8)
n_valid = len(ds_iris) - n_train
ds_train, ds_valid = random_split(ds_iris, [n_train, n_valid])

print(type(ds_iris))
print(type(ds_train))

dl_train, dl_valid = DataLoader(
    ds_train, batch_size=8), DataLoader(ds_valid, batch_size=8)

for features, labels in dl_train:
    print(features, labels)
    break

ds_data = ds_train + ds_valid

print(len(ds_train))
print(len(ds_valid))
print(len(ds_data))

# 2.根据图片目录创建图片数据集

# 自定义图片增强操作
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
])

ds_train = datasets.ImageFolder('./data/cifar2/train',
                                transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
ds_valid = datasets.ImageFolder('./data/cifar2/test',
                                transform=transform_valid, target_transform=lambda t: torch.tensor([t]).float())
print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=3)
dl_valid = DataLoader(ds_valid, batch_size=50, shuffle=True, num_workers=3)

for features, labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break

# 3.自定义数据集
# 使用自定义数据集加载imdb文本分类数据
# 还没学习到文本处理，所以这一部分内容先略过

# 使用DataLoader加载数据集
# Dataloader的主要作用是控制数据集的读入方式，例如batch大小，采样方法，输出数据格式，以及配置多进程读取数据

# DataLoader类各参数含义
# dataset 数据集
# batch_size 批次大小
# shuffle 是否乱序
# num_worker 读取进程数量
# collate_fn 批次整理函数
# pin_memory 是否使用虚拟内存，默认False
# drop_last 是否丢弃最后一组不完整的批次
# timeout 加载一个批次最长的等待时间，一般默认
# worker_init_fn 加载器函数，一般默认
# sampler 样本采样函数，一般默认
# batch_sampler 批次采样函数，一般默认

ds = TensorDataset(torch.arange(1, 50))
dl = DataLoader(ds,
                batch_size=10,
                shuffle=True,
                num_workers=2,
                drop_last=True)
for batch in dl:
    print(batch)
    break
