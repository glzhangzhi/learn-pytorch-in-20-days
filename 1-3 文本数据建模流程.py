# 一。数据准备

# 在torch中预处理文本数据一般使用torchtext或者自定义Dataset，torchtext功能非常强大，可以构建文本分类，序列标注
# 问答模型，机器翻译等NLP任务的数据集

'''
torchtext常用API
torchtext.data.Example 用来表示一个样本的数据和标签
torchtext.vocab.Vocab 词汇表，可以导入一些预训练词向量
torchtext.data.Datasets 数据集类，__getitem__返回Example实例，torchtext.data.TabularDatasets是其子类
torchtext.data.Field 用来定义字段的处理方法（文本字段、标签字段），创建Example时的预处理，生成batch时的一些处理操作
torchtext.data.Iterator 迭代器，用来生成batch
torchtext.datasets 包含了常用的数据集
'''

import torch
import string, re
import torchtext

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 100  # 每个样本保留200个词的长度
BATCH_SIZE = 20

# 分词方法
tokenizer = lambda x:re.sub('[%s]'%string.punctuation,'',x).split(' ')

# 过滤低频词
def filterLowFreqWords(arr, vocab):
    arr = [[x if x<MAX_WORDS else 0 for x in Example] for example in arr]
    return arr

