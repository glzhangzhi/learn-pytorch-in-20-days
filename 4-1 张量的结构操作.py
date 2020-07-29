import torch

# 创建张量

a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.arange(1, 10, step=2)
c = torch.linspace(0.0, 2 * 3.14, 10)
d = torch.zeros((3, 3))
e = torch.ones((3, 3), dtype=torch.int)
f = torch.zeros_like(e, dtype=torch.float)
torch.fill_(f, 5)
torch.manual_seed(0)
minval, maxval = 0, 10
g = minval + (maxval - minval) * torch.rand([5])  # 随机均匀分布
h = torch.normal(mean=torch.zeros(3, 3), std=torch.ones(3, 3))  # 随机正态分布
i = torch.eye(3, 3)  # 单位矩阵
j = torch.diag(torch.tensor([1, 2, 3]))  # 对角矩阵

# 索引切片

a[0]
a[-1]
a[1, 3]  # g[1][3]
a[1:4, :]
a[1:4, :4:2]
a.data[1, :] = torch.tensor([0.0, 0.0])
a = torch.arange(27).view(3, 3, 3)

# 维度变换

a_321 = torch.reshape(a, [3, 2, 1])
a_32 = torch.squeeze(a_321)
a_312 = torch.unsqueeze(a_32, axis=1)
a_321 = torch.transpose(a_312, 1, 2)

# 合并分割

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

abc_cat = torch.cat([a, b, c], dim=0)  # 将abc的元素堆叠
abc_stack = torch.stack([a, b, c], axis=0)  # 将abc作为元素堆叠
a, b, c = torch.split(abc_cat, split_size_or_sections=2, dim=0)  # 每份2个进行分割
p, q, r = torch.split(abc_cat, split_size_or_sections=[
                      4, 1, 1], dim=0)  # 每份分别为[4,1,1]
