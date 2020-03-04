import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # 加载数据集将返回四个NumPy数组
# 每个图像都映射到一个标签。由于类名不包含在数据集中，因此将它们存储在此处以供以后在绘制图像时使用：
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
"""
在训练模型之前，让我们探索数据集的格式。下图显示了训练集中有60,000张图像，每个图像表示为28 x 28像素：
(60000, 28, 28)
同样，训练集中有60,000个标签：
60000
每个标签都是0到9之间的整数：
[9 0 0 ... 3 0 5]
测试集中有10,000张图像。同样，每个图像都表示为28 x 28像素：
(10000, 28, 28)
测试集包含10,000个图像标签：
10000
"""
# 在训练网络之前，必须对数据进行预处理。如果检查训练集中的第一张图像，您将看到像素值落在0到255之间：
plt.figure()
plt.imshow(train_images[0])  # plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示
plt.colorbar()
plt.grid(False)
plt.show()  # 使用plt.show()显示出来
"""
将这些值缩放到0到1的范围，然后再将其输入神经网络模型。为此，将值除以255。
以相同的方式预处理训练集和测试集非常重要
为了验证数据的格式正确，并准备好构建和训练网络，让我们显示训练集中的前25个图像，并在每个图像下方显示类别名称
"""
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # 显示黑白图片
    plt.xlabel(class_names[train_labels[i]])
plt.show()
