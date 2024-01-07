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
        images.append(image)
    return np.array(images)


mnist_images = load_images_and_labels('dataset/MNIST/mnist_dataset/train')
color_images = load_images_and_labels('dataset/MNIST_color/testset/img')

y_binary = np.loadtxt('dataset/MNIST/mnist_dataset/less_train_labs.txt',dtype=np.int64) # less_train_labs
y_color = np.loadtxt('dataset/MNIST_color/testset/test_labs.txt',dtype=np.int64)

X_train = torch.tensor(mnist_images[y_binary[:,0]].reshape(-1, 1, 28, 28),dtype=torch.float32) # [y_binary[:,0]]
y_train = torch.tensor(y_binary[:,1],dtype=torch.long)

X_test = torch.tensor(color_images.reshape(-1, 1, 28, 28),dtype=torch.float32)
y_test = torch.tensor(y_color[:,1],dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)
 
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
        output = f.log_softmax(x, dim=1)
        return output
 
 
learn_rate = 0.01 
model = Model()
model = model.cuda()
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate)

 
if __name__ == '__main__':
    epoch = 300
    for i in range(epoch):
        correct = 0 
        total = 0 

        # train
        for _, data in enumerate(train_loader):
            input, target = data
            input = input.cuda()
            target = target.cuda()
            y_predict = model(input)
            loss = criterion(y_predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # test
        with torch.no_grad(): 
            for data in test_loader:
                input, target = data
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                probability, predict = torch.max(output.data, dim=1)
                total += target.size(0)
                correct += (predict == target).sum().item()
            print(f"第{i+1}轮，模型测试时准确率为: %.4f" % (correct / total))
