import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from torchvision import transforms
import random
from PIL import Image
import torch

def load_images_and_labels(dataset_path):
    images = []
    files = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0]))
    for file in files:
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (28, 28)) 
        images.append(image)
    return np.array(images)

# 加载MNIST数据集
mnist_images = load_images_and_labels('dataset/MNIST/mnist_dataset/train')

# 加载MNIST_color数据集
color_images = load_images_and_labels('dataset/MNIST_color/testset/img')

# 合并两个数据集
# all_images = np.concatenate((mnist_images, color_images))
# all_labels = np.concatenate((mnist_labels, color_labels))

y_binary = np.loadtxt('dataset/MNIST/mnist_dataset/less_train_labs.txt',dtype=np.int64) # train_labs less_train_labs
y_color = np.loadtxt('dataset/MNIST_color/testset/test_labs.txt',dtype=np.int64)
# 划分数据集

# X_train = mnist_images[y_binary[:,0].astype(int)].reshape(-1, 28*28)
# y_train = y_binary[:,1]

# X_test = color_images.reshape(-1, 28*28)
# y_test = y_color[:,1]

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
        self.trans_list.append(transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.9, 1.1), shear=(0, 0)))
        # self.trans_list.append(transforms.RandomPerspective(distortion_scale=0.3, p=1.1, interpolation=Image.BICUBIC))
        # self.trans_list.append(transforms.RandomAffine(degrees=0, scale=(1.0, 1.0), shear=(-3, 3)))

    def __call__(self, image_tensor):
        image_pil = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
        # transformed_image_pil = self.transforms(image_pil)
        transformed_image_pil = self.trans_list[random.randint(0,len(self.trans_list)-1)](image_pil)
        return transforms.ToTensor()(transformed_image_pil).unsqueeze(0)

X_train = torch.tensor(mnist_images[y_binary[:,0]].reshape(-1, 1, 28, 28),dtype=torch.float32)
y_train = torch.tensor(y_binary[:,1],dtype=torch.long)
X_test = torch.tensor(color_images.reshape(-1, 1, 28, 28),dtype=torch.float32)
y_test = torch.tensor(y_color[:,1],dtype=torch.long)

custom_transforms = CustomTransforms()
for i in range(0, 2001):
    # idx = random.randint(0, X_train.shape[0]-1)
    idx = i
    img = X_train[idx]
    y_idx = y_train[idx].unsqueeze(0)
    X_train = torch.cat([X_train, custom_transforms(img)], dim=0)
    y_train = torch.cat([y_train, y_idx], dim=0)

X_train = X_train.squeeze().view(-1,784).numpy()
X_test = X_test.squeeze().view(-1,784).numpy()
y_train = y_train.squeeze().numpy()
y_test = y_test.squeeze().numpy()
print(X_train.shape[0], y_train.shape[0])
# 创建SVM模型
svm_model = svm.SVC(kernel='rbf')

# 训练模型
svm_model.fit(X_train, y_train)

# 预测
y_pred_svm = svm_model.predict(X_test)

# 评估准确性
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")
