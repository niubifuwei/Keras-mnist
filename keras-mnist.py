from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#定义损失函数、优化器及评价指标
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


#mnist数据的预处理。
train_images = train_images.reshape((60000, 28 * 28))  #把一个图像变成一列数据用于学习,其形状应该与上面input_shape相同，因此做reshape操作
train_images = train_images.astype('float32') / 255 #astype用于进行数据类型转换

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#训练 
from keras.utils import to_categorical

train_labels=to_categorical(train_labels)#类别向量转换为二进制（只有0和1）的矩阵类型表示
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

digit = train_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()