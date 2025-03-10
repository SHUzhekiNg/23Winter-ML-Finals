import os
import cv2
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
import tqdm

def load_images_and_labels(dataset_path):
    images = []
    files = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0]))
    for file in files:
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (28, 28)) 
        images.append(image)
    return np.array(images)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)#.to(torch.int32)
        # output = torch.log_softmax(output, dim=1)
        return output

# 加载MNIST数据集
mnist_images = load_images_and_labels('dataset/MNIST/mnist_dataset/train')

# 加载MNIST_color数据集
# color_images = load_images_and_labels('dataset/MNIST_color/testset/img')
color_images = load_images_and_labels('dataset/MNIST/mnist_dataset/test')
# 合并两个数据集
# all_images = np.concatenate((mnist_images, color_images))
# all_labels = np.concatenate((mnist_labels, color_labels))

y_binary = np.loadtxt('dataset/MNIST/mnist_dataset/train_labs.txt') # less_train_labs
# y_color = np.loadtxt('dataset/MNIST_color/testset/test_labs.txt')
y_color = np.loadtxt('dataset/MNIST/mnist_dataset/test_labs.txt')
# 划分数据集

X_train = torch.tensor(mnist_images[y_binary[:,0].astype(int)].reshape(-1, 1, 28, 28),dtype=torch.float32)
y_train = torch.tensor(y_binary[:,1],dtype=torch.float32)

X_test = torch.tensor(color_images.reshape(-1, 1, 28, 28),dtype=torch.float32)
y_test = torch.tensor(y_color[:,1],dtype=torch.float32)

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型、损失函数和优化器
lr = 0.01
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs, labels.to(torch.long))
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')
