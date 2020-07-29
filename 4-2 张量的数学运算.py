# 标量运算（逐个元素运算）

import torch

a = torch.tensor([[1.0, 2], [-3, 4.0]])
b = torch.tensor([[5.0, 6], [7.0, 8.0]])

a + b
a - b
a * b
a / b
a ** 2
a ** 0.5
a % 5
a >= 2  # torch.ge(a, 2) greater_equal
(a >= 2) & (a <= 3)
(a >= 2) | (a <= 3)
a == 5  # torch.eq(a, 5)
torch.sqrt(a)
torch.max(a, b)  # 分别球各列上最大的值
torch.min(a, b)
torch.round(a)  # 四舍五入取整
torch.floor(a)  # 向下取整
torch.cell(a)  # 向上取整
torch.trunc(a)  # 向0取整
torch.fmod(a, 2)  # 求余
torch.clamp(a, min=-1, max=1)   # 幅值裁减

# 向量运算

a = torch.arange(1, 10).float()

torch.sum(a)
torch.sum(a, dim=0)
torch.mean(a)
torch.max(a)
torch.min(a)
torch.prod(a)  # 累乘
torch.std(a)  # 标准差
torch.var(a)  # 方差
torch.median(a)  # 中位数

# 矩阵运算(二维向量)

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[2, 0], [2, 0]])

a.t()
torch.inverse(a)
