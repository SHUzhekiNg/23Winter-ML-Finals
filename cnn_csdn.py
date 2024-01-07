# minist_all_GPU.py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as f
 
# 训练数据集
train_data = MNIST(root='./MNIST', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64)
 
# 测试数据集
test_data = MNIST(root='./MNIST', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64)
 
# 观察训练数据集、测试数据集中的图像有多少张
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))  # 训练数据集的长度为：60000
print("测试数据集的长度为：{}".format(test_data_size))  # 测试数据集的长度为：10000
 

# 模型
class Model(nn.Module):
    """
    编写一个卷积神经网络类
    """
 
    def __init__(self):
        """ 初始化网络,将网络需要的模块拼凑出来。 """
        super(Model, self).__init__()
        # 卷积层:
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        # 最大池化处理:
        self.pooling = nn.MaxPool2d(2, 2)
        # 全连接层：
        self.fc1 = nn.Linear(16 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        """前馈函数"""
        x = f.relu(self.conv1(x))  # = [b, 6, 28, 28]
        x = self.pooling(x)  # = [b, 6, 14, 14]
        x = f.relu(self.conv2(x))  # = [b, 16, 14, 14]
        x = self.pooling(x)  # = [b, 16, 7, 7]
        x = x.view(x.shape[0], -1)  # = [b, 16 * 7 * 7]
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output
 
 
# CrossEntropyLoss
learn_rate = 0.01  # 学习率
model = Model()  # 模型实例化
model = model.cuda() # 使用GPU
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，相当于Softmax+Log+NllLoss
criterion = criterion.cuda() # 使用GPU
optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate)  # 第一个参数是初始化参数值，第二个参数是学习率
 
 
 
# 模型训练
def train():
    # 记录训练的次数 训练次数——每次训练一个batch_size（64）的图片，每轮要训练60000/64次
    total_train_step = 0
    batch_losses = [] # 存放每batch训练后的损失
    step = [] # 损失曲线x轴的间隔
    for index, data in enumerate(train_loader):  # index表示data的索引   或者 for data in train_loader:
        input, target = data  # input为输入数据，target为标签
        # 使用GPU
        input = input.cuda()
        target = target.cuda()
 
        y_predict = model(input)  # 模型预测
        loss = criterion(y_predict, target)  # 计算损失
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_train_step = total_train_step + 1
 

        if total_train_step % 64 == 0:  # 每一个batch_size打印损失
            # print("训练次数:{},模型训练时的损失值为:{}".format(total_train_step, loss.item()))
            batch_losses.append(loss)
            step.append(total_train_step)
    return batch_losses, step
 
# 模型测试
def test(i):
    correct = 0  # 正确预测的个数
    total = 0  # 总数
    with torch.no_grad():  # 测试不用计算梯度
        for data in test_loader:
            input, target = data
            # 使用GPU
            input = input.cuda()
            target = target.cuda()
 
            output = model(input)  # output输出10个预测取值，其中最大的即为预测的数
            probability, predict = torch.max(output.data, dim=1)  # 返回一个元组，第一个为最大概率值，第二个为最大值的下标
            total += target.size(0)  # target是形状为(batch_size,1)的矩阵，使用size(0)取出该批的大小
            correct += (predict == target).sum().item()  # predict和target均为(batch_size,1)的矩阵，sum()求出相等的个数
        print(f"第{i+1}轮，模型测试时准确率为: %.4f" % (correct / total))
 
 
epoch = 50  # 训练轮数 训练轮数——每轮训练整体60000张图片，轮数越多，模型准确率越高
for i in range(epoch):  # 训练和测试进行5轮
    # print("———————第{}轮训练开始——————".format(i + 1))
    batch_losses, step = train() 
    # 模型测试
    test(i)