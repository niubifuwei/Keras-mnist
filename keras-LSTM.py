import keras
from keras.layers import LSTM
from keras.layers import Dense,Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

learning_rate=0.001
training_iters=10
batch_size=128
display_step=10

n_input=28
n_step=28
n_hidden=128
n_classes=10

f=np.load('/home/tianchi/myspace/mnist.npz')
x_train,y_train=f['x_train'],f['y_train']
x_test,y_test=f['x_test'],f['y_test']

print(len(x_train))
#print(x_train[0])
print(x_train.shape)
print(type(x_train))

x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(len(x_train))
#print(x_train[0])
print(x_train.shape)
print(type(x_train))

#
x_train /= 255
x_test /= 255

#print(x_train[0])

print(y_train[0])
#把整形int标签转换成one-hot编码的数组标签,以方便计算loss
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
print(y_train[0])

#搭建模型
model=Sequential()
model.add(LSTM(n_hidden,batch_input_shape=(None,n_step,n_input),unroll=True))
model.add(Dense(n_classes))#这个参数应该与输出维度相同了
model.add(Activation('softmax'))

adam=Adam(lr=learning_rate)
model.summary()
model.compile(optimizer=adam,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=batch_size,epochs=training_iters,verbose=1,validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)
print('LSTM test score:',score[0])
print('LSTM test accuracy:',score[1])