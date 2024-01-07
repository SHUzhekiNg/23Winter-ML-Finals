import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_images_and_labels(dataset_path):
    images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28)) 
            images.append(image)

    return np.array(images)

# 加载MNIST数据集
mnist_images = load_images_and_labels('dataset/MNIST/mnist_dataset/train')

# 加载MNIST_color数据集
color_images = load_images_and_labels('dataset/MNIST_color/testset/img')

# 合并两个数据集
# all_images = np.concatenate((mnist_images, color_images))
# all_labels = np.concatenate((mnist_labels, color_labels))

y_binary = np.loadtxt('dataset/MNIST/mnist_dataset/less_train_labs.txt')
y_color = np.loadtxt('dataset/MNIST_color/testset/test_labs.txt')
# 划分数据集

X_train = mnist_images[y_binary[:,0].astype(int)].reshape(-1, 28*28)
y_train = y_binary[:,1]

X_test = color_images.reshape(-1, 28*28)
y_test = y_color[:,1]
# 创建SVM模型
svm_model = svm.SVC(kernel='linear')

# 训练模型
svm_model.fit(X_train, y_train)

# 预测
y_pred_svm = svm_model.predict(X_test)

# 评估准确性
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")
