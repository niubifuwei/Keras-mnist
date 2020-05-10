from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input,Dense,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D,GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os
import numpy as np

class DCGAN():
    def __init__(self):
        #输入shape
        self.img_rows=28
        self.img_cols=28
        self.channels=1
        self.img_shape=(self.img_rows,self.img_cols,self.channels)
        #分十类
        self.nun_classes=10
        self.latent_dim=100
        #adam优化器
        optimizer=Adam(0.0002,0.5)
        #判别模型
        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])
        #生成模型
        self.generator=self.build_generator()
        
        #Conbine是生成模型和判别模型的结合
        #判别模型的trainable为False
        #用于训练生成模型
        z=Input(shape=(self.latent_dim,))
        img=self.generator(z)
        
        self.discriminator.trainable=False
        valid=self.discriminator(img)
        self.combined=Model(z,valid) #输入z,得到一个（0-1）的概率
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer) #定义其损失函数
        
    def build_generator(self):
        model=Sequential()
        #先全连接到32*7*7的维度上
        model.add(Dense(32*7*7,activation='relu',input_dim=self.latent_dim))
        #reshape成特征层的样式
        model.add(Reshape((7,7,32)))
        
        #7,7,64
        model.add(Conv2D(64,kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=0.8))#还没搞懂
        model.add(Activation('relu'))
        #上采样
        #7,7,64->14,14,64
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        # 上采样
        # 14, 14, 128 -> 28, 28, 64
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # 上采样
        # 28, 28, 64 -> 28, 28, 1
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        
        model.summary()
        
        noise=Input(shape=(self.latent_dim,))
        img=model(noise)
        return Model(noise,img)
    
    def build_discriminator(self):
        model=Sequential()
        #28,28,1->14,14,32
        model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=self.img_shape,padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
        #14,14,32->7,7,64
        model.add(Conv2D(64,kernel_size=3,strides=2,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        #7,7,64->4,4,4,128
        model.add(ZeroPadding2D(((0,1),(0,1))))
        model.add(Conv2D(128,kernel_size=3,strides=2,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GlobalAveragePooling2D())#GlobalAveragePooling2D中的运算是K.mean(input, axis=[1, 2])
        
        #全连接
        model.add(Dense(1,activation='sigmoid'))#输出一个[0-1]的概率以表示是否是真实数据
        model.summary()
        
        img=Input(shape=self.img_shape)
        validity=model(img)
        
        return Model(img,validity)#图片，有效性
    
    def train(self,epochs,batch_size=128,save_interval=50):
        #载入数据
        f=np.load('/home/tianchi/myspace/mnist.npz')
        x_train,y_train=f['x_train'],f['y_train']
        x_test,y_test=f['x_test'],f['y_test']
        
        #归一化
        x_train=x_train/127.5-1
        x_train=np.expand_dims(x_train,axis=3)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))#生成batch_size行 1 列的矩阵   真的
        fake = np.zeros((batch_size, 1)) #假的
        
        for epoch in range(epochs):
            
            #训练判别模型
            idx=np.random.randint(0,x_train.shape[0],batch_size)#随机生成batch_size个[0,60000]
            imgs=x_train[idx] #随机生成batch_size个真实图片
            
            noise=np.random.normal(0,1,(batch_size,self.latent_dim))
            gen_imgs=self.generator.predict(noise)#通过生成器生成的图片
            
            #训练并计算loss
            d_loss_real=self.discriminator.train_on_batch(imgs,valid)#真实数据和真实数据的预测值
            d_loss_fake=self.discriminator.train_on_batch(gen_imgs,fake)#假数据和假数据的预测值
            d_loss=0.5*np.add(d_loss_real,d_loss_fake)
            
            #训练生成模型
            g_loss=self.combined.train_on_batch(noise,valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            if epoch % save_interval==0:
                self.save_imgs(epoch)
                
    def save_imgs(self,epoch):
        r,c=5,5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs=self.generator.predict(noise)
        gen_imgs=0.5*gen_imgs+0.5
        
        fig,axs=plt.subplots(r,c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0],cmap='gray')
                axs[i,j].axis('off')
                cnt+=1
        fig.savefig("/home/tianchi/myspace/images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("/home/tianchi/myspace/images"):
        os.makedirs("/home/tianchi/myspace/images")
    dcgan = DCGAN()
    dcgan.train(epochs=2000, batch_size=256, save_interval=50)
                
            
        
    
    
        

