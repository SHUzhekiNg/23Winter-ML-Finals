# minist_all_GPU.py
import numpy as np
import torch
import os
import cv2
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as f
import random
 
def load_images_and_labels(dataset_path):
    images = []
    files = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0]))
    for file in files:
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (24, 24)) 
        images.append(image)
    return np.array(images)


def add_data(X, y):
    # 0 for rotation 90, 180, 270
    indices = (y == 0).nonzero().squeeze()
    X_train_y0 = X[indices]
    X_train_y0_rotated1 = torch.rot90(X_train_y0, k=1, dims=(2, 3))
    X_train_y0_rotated2 = torch.rot90(X_train_y0, k=2, dims=(2, 3))
    X_train_y0_rotated3 = torch.rot90(X_train_y0, k=-1, dims=(2, 3))
    X = torch.cat([X, X_train_y0_rotated1, X_train_y0_rotated2, X_train_y0_rotated3], dim=0)
    y = torch.cat([y, torch.zeros((3*indices.shape[0]), dtype=torch.long)], dim=0)

    # # 6 for rotation 180 as 9
    # indices = (y == 6).nonzero().squeeze()
    # X_train_y0 = X[indices]
    # X_train_y0_rotated2 = torch.rot90(X_train_y0, k=2, dims=(2, 3))
    # X = torch.cat([X, X_train_y0_rotated2], dim=0)
    # y = torch.cat([y, torch.ones((indices.shape[0]), dtype=torch.long)*9], dim=0)

    # # 9 for rotation 180 as 6
    # indices = (y == 9).nonzero().squeeze()
    # X_train_y0 = X[indices]
    # X_train_y0_rotated2 = torch.rot90(X_train_y0, k=2, dims=(2, 3))
    # X = torch.cat([X, X_train_y0_rotated2], dim=0)
    # y = torch.cat([y, torch.ones((indices.shape[0]), dtype=torch.long)*6], dim=0)

    # 8 for rotation 180 as 8
    indices = (y == 8).nonzero().squeeze()
    X_train_y0 = X[indices]
    X_train_y0_rotated2 = torch.rot90(X_train_y0, k=2, dims=(2, 3))
    X = torch.cat([X, X_train_y0_rotated2], dim=0)
    y = torch.cat([y, torch.ones((indices.shape[0]), dtype=torch.long)*8], dim=0)

    # # 1 for rotation 180 as 1
    # indices = (y == 1).nonzero().squeeze()
    # X_train_y0 = X[indices]
    # X_train_y0_rotated2 = torch.rot90(X_train_y0, k=2, dims=(2, 3))
    # X = torch.cat([X, X_train_y0_rotated2], dim=0)
    # y = torch.cat([y, torch.ones((indices.shape[0]), dtype=torch.long)], dim=0)
    return X, y

class CustomTransforms:
    def __init__(self):
        # self.transforms = transforms.Compose([
        #     transforms.RandomApply([transforms.RandomRotation(degrees=(-10, 10))], p=0.25), # , resample=Image.BICUBIC
        #     transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(1.0, 1.0), shear=(-2, 2))], p=0.25),
        #     transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.3, p=1.0, interpolation=Image.BICUBIC)], p=0.25),
        #     transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(1.0, 1.0), shear=(-3, 3))], p=0.25),
        # ])
        self.trans_list = []
        # self.trans_list.append(transforms.RandomRotation(degrees=(-5, 5)))
        # self.trans_list.append(transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.9, 1.1), shear=(0, 0)))
        self.trans_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=1.1, interpolation=Image.BICUBIC))
        # self.trans_list.append(transforms.RandomAffine(degrees=0, scale=(1.0, 1.0), shear=(-3, 3)))

    def __call__(self, image_tensor):
        image_pil = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        # transformed_image_pil = self.transforms(image_pil)
        transformed_image_pil = self.trans_list[random.randint(0,len(self.trans_list)-1)](image_pil)
        return transforms.ToTensor()(transformed_image_pil).unsqueeze(0)

    
mnist_images = load_images_and_labels('dataset/MNIST/mnist_dataset/train')
color_images = load_images_and_labels('dataset/MNIST_color/testset/img')

y_binary = np.loadtxt('dataset/MNIST/mnist_dataset/less_train_labs.txt',dtype=np.int64) # less_train_labs
y_color = np.loadtxt('dataset/MNIST_color/testset/test_labs.txt',dtype=np.int64)

X_train = torch.tensor(mnist_images[y_binary[:,0]].reshape(-1, 1, 24, 24),dtype=torch.float32)
y_train = torch.tensor(y_binary[:,1],dtype=torch.long)

X_test = torch.tensor(color_images.reshape(-1, 1, 24, 24),dtype=torch.float32)
y_test = torch.tensor(y_color[:,1],dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 创建自定义的图像变换对象
# custom_transforms = CustomTransforms()
# for i in range(0, 2001):
#     # idx = random.randint(0, X_train.shape[0]-1)
#     img = X_train[i]
#     y_idx = y_train[i].unsqueeze(0)
#     X_train = torch.cat([X_train, custom_transforms(img)], dim=0)
#     y_train = torch.cat([y_train, y_idx], dim=0)

# HOG特征提取
def hog_features(images):
    hog_features_list = []
    hog = cv2.HOGDescriptor((24, 24), (12, 12), (2, 2), (4, 4), 9)
    for img in images:
        img = np.reshape(img, (24, 24)).astype(np.uint8)
        hog_features = hog.compute(img)
        hog_features_list.append(hog_features.flatten())
    return np.array(hog_features_list)

X_train_hog = hog_features(X_train.numpy()).reshape(-1, 1, 63, 63)# (-1, 1, 18, 18)
X_test_hog = hog_features(X_test.numpy()).reshape(-1, 1, 63, 63)
X_train = torch.tensor(X_train_hog, dtype=torch.float32)
X_test = torch.tensor(X_test_hog, dtype=torch.float32)
# X_train, y_train = add_data(X_train, y_train)
# X_train = torch.cat([X_train, X_train], dim=0)
# y_train = torch.cat([y_train, y_train], dim=0)
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
 
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
        self.fc1 = nn.Linear(32 * 15 * 15, 512) # 32*7*7 32*4*4 32*12*12
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
 
 
learn_rate = 0.02
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

    print("----EVAL ALL----")
    # 初始化每个类别的样本数和正确分类的样本数
    class_correct = [0] * 10
    class_total = [0] * 10

    # test
    with torch.no_grad(): 
        correct = 0 
        total = 0 
        for data in test_loader:
            input, target = data
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            probability, predict = torch.max(output.data, dim=1)
            
            # 统计每个类别的样本数和正确分类的样本数
            total += target.size(0)
            correct += (predict == target).sum().item()

            for i in range(10):
                class_total[i] += (target == i).sum().item()
                class_correct[i] += (predict == target)[target == i].sum().item()

    # 输出每个类别的准确率
    for i in range(10):
        print(f"类别 {i} 的准确率为: %.4f" % (class_correct[i] / class_total[i]))