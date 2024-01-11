import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
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
        image = cv2.resize(image, (32, 32)) 
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

X_train = mnist_images[y_binary[:,0].astype(int)].reshape(-1, 32*32)
y_train = y_binary[:,1]

X_test = color_images.reshape(-1, 32*32)
y_test = y_color[:,1]

# HOG特征提取
def hog_features(images):
    hog_features_list = []
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (4, 4), (4, 4), 9)
    for img in images:
        img = np.reshape(img, (32, 32))
        hog_features = hog.compute(img)
        hog_features_list.append(hog_features.flatten())
    return np.array(hog_features_list)

X_train_hog = hog_features(X_train)
X_test_hog = hog_features(X_test)

# 创建SVM模型
svm_model = svm.SVC(kernel='rbf')

# 训练模型
svm_model.fit(X_train_hog, y_train)

# 预测
y_pred_svm = svm_model.predict(X_test_hog)

# 评估准确性
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")

class_accuracy = classification_report(y_test, y_pred_svm, digits=4)
print(f"Class-wise Accuracy:\n{class_accuracy}")