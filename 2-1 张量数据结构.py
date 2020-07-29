# pytorch的基本数据结构是张量Tensor，即多维数组，与numpy中的array很相似。


# 一。张量的数据类型

# 除了str，涵盖了基本的数据类型
# torch.float64(torch.double),

# torch.float32(torch.float),

# torch.float16,

# torch.int64(torch.long),

# torch.int32(torch.int),

# torch.int16,

# torch.int8,

# torch.uint8,

# torch.bool

# 一般神经网络建模使用的都是torch.float32类型


import numpy as np
import torch

# 自动推断数据类型
i = torch.tensor(1)
x = torch.tensor(2.0)
b = torch.tensor(True)

# 指定数据类型
i = torch.tensor(1, dtype=torch.int32)
x = torch.tensor(2.0, dtype=torch.double)

# 使用特定类型的构造函数
i = torch.IntTensor(1)
x = torch.Tensor(np.array(2.0))  # 等价于torch.FloatTensor
b = torch.BoolTensor(np.array[1, 0, 2, 0])

# 不同类型转换
i = torch.tensor(1)
x = i.float()
y = i.type(torch.float)
z = i.type_as(x)

# 二。张量的维度
# 简单来说，有几层中括号，就有几层张量

scalar = torch.tensor(True)
print(scalar.dim())  # 0维就是标量

vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector.dim())  # 1维就是向量

matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(matrix.dim())  # 2维就是矩阵

tensor3 = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])
print(tensor3.dim())

# 三。张量的尺寸
# 使用shape属性或size()方法查看张量在每一维的长度
# 使用view()方法改变张量的尺寸
# 如果view()方法失败，可以使用reshape()方法
# 有些操作会让张量储存结构扭曲，比如转置，直接使用view会失败，可以使用reshape方法

scalar.size()
vector.shape
matrix.size

matrix34 = vector.view(3, 4)
matrix43 = vector.view(4, -1)

matrix26 = torch.arange(0, 12).view(2, 6)
matrix62 = matrix26.t()

matrix34 = matrix62.view(3, 4)  # 这个会报错
matrix34 = matrix62.reshape(3, 4)
# 等价于
matrix34 = matrix62.contiguous().view(3, 4)

# 四。张量和numpy数组
# numpy的array和pytorch的tensor可以相互转换
# 这样转换来的对象是共享数据内存的，改变其中一个，另一个也会改变
# 可以使用张量的clone方法拷贝，中断这种共享关系
# 此外，还可以使用item方法从标量张量中得到对应的python数值
# 使用tolist()从张量得到对应的python数值列表

import numpy as np
import torch

# 转换后内存关联
arr = np.zeros(3)
tensor = torch.from_numpy(arr)
np.add(arr, 1, out=arr)

# 使用clone断开内存关联
tensor = torch.zeros(3)
arr = tensor.clone().numpy()
# 也可以使用tensor.data.numpy()

# 取得单个数据
scalar = torch.tensor(1.0)
s = scalar.item()

# 取得列表数据
tensor = torch.rand(2, 2)
t = tensor.tolist()
