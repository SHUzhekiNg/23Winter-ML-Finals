# minist_all_GPU.py
import numpy as np
import torch
import os
import cv2
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as f
 
def load_images_and_labels(dataset_path):
    images = []
    files = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0]))
    for file in files:
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (28, 28)) 
        images.append(image)
    return np.array(images)

mnist_images = load_images_and_labels('dataset/MNIST/mnist_dataset/train')
color_images = load_images_and_labels('dataset/MNIST_color/testset/img')

y_binary = np.loadtxt('dataset/MNIST/mnist_dataset/train_labs.txt',dtype=np.int64) # train_labs less_train_labs
y_color = np.loadtxt('dataset/MNIST_color/testset/test_labs.txt',dtype=np.int64)

X_train = torch.tensor(mnist_images.reshape(-1, 1, 28, 28),dtype=torch.float32) # [y_binary[:,0]]
y_train = torch.tensor(y_binary[:,1],dtype=torch.long)

X_test = torch.tensor(color_images.reshape(-1, 1, 28, 28),dtype=torch.float32)
y_test = torch.tensor(y_color[:,1],dtype=torch.long)

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)
 
# 观察训练数据集、测试数据集中的图像有多少张
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))  # 训练数据集的长度为：60000
print("测试数据集的长度为：{}".format(test_data_size))  # 测试数据集的长度为：10000
 

# 模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        # self.pooling = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(16 * 7 * 7, 512)
        # self.fc2 = nn.Linear(512, 10)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pooling(x)
        x = f.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        output = x #f.log_softmax(x, dim=1)
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