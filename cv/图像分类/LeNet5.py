import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import cv2


def LeNet():
    input_image = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(6, kernel_size=5, padding="same", activation="sigmoid")(input_image)
    x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=5, activation="sigmoid")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(120, kernel_size=5, activation="sigmoid")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(84, activation="sigmoid")(x)
    x = tf.keras.layers.Dense(10, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=input_image, outputs=x)
    return model

def train(data_path,model_path):

    '''数据处理
    目标：原数据->模型能读入的数据(分为训练集和测试集，细分为输入数据和输出结果，再细分为batch)
    '''
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(
        path=data_path)
    # 维度变换+归一化
    train_images = train_images.reshape(60000, 28, 28, 1) / 255
    test_images = test_images.reshape(10000, 28, 28, 1) / 255
    # 对标签进行分类编码(one-hot编码)
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    '''模型定义
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 6,kernel_size = (5,5),input_shape=(28,28,1),padding = 'same',activation = "sigmoid"))
    #padding='same',设置卷积后，输入图片相对于输出图片尺寸不变，tf自动给设置padding的值；padding='valid'(丢弃)，strides=1
    model.add(tf.keras.layers.AveragePooling2D(pool_size = (2, 2)))
    model.add(tf.keras.layers.Conv2D(filters = 16,kernel_size = (5,5),activation = "sigmoid"))#卷积核生成特征图
    model.add(tf.keras.layers.AveragePooling2D(pool_size = (2, 2)))#池化是对特征图池化
    model.add(tf.keras.layers.Conv2D(filters = 120,kernel_size = (5,5),activation = "sigmoid"))#卷积过程就是提取特征过程
    model.add(tf.keras.layers.Flatten())#展平层
    model.add(tf.keras.layers.Dense(84, activation='sigmoid'))#神经网络前向传播，sigmoid激活函数
    model.add(tf.keras.layers.Dense(10, activation='softmax'))#softmax线性分类器，将像素值转化为得分再转化为概率
    '''或者这样定义
    model = LeNet()
    '''
    #模型层数预览
    model.summary()


    '''模型导入
    导入模型，设置优化器，损失函数
    '''
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

    '''训练
    训练epoch轮
    '''
    tensorboard=TensorBoard(log_dir='./lenet5_logs',write_graph=True)
    #adam优化器，反向传播梯度下降自动改变步长，自己设置的话就是0.0005；损失函数为交叉熵函数，计算预测的损失值；打印预测值
    history = model.fit(train_images,train_labels,epochs = 10,validation_data=(test_images,test_labels),callbacks=[(tensorboard)])
    #载入训练集、验证级、设置训练轮次；验证集随便选，和训练效果无关

    '''训练完毕进行评估
    可以根据各种指标计算模型性能，并保存模型
    '''
    model.evaluate(test_images,test_labels)
    model.save(model_path)


def evaluate(model_path,image_path):
    '''导入模型
    '''
    new_model = tf.keras.models.load_model(model_path)

    '''读取数据，转化成模型可以输入的格式
    '''
    img = cv2.imread(image_path, 0)
    # 读取图片,0是灰度图，1是彩色图
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    img = img / 255

    '''调用模型得到结果
    '''
    predict = new_model.predict(img)

    '''对结果做处理，得到想要的数据
    '''
    res = np.argmax(predict)
    print(res)


if __name__=="__main__":
    # data_path="F://zzx//tensorflow_data//mnist.npz"
    # model_path='saved_model/my_model2/LeNet5.h5'
    # train(data_path=data_path,model_path=model_path)

    model_path='saved_model/my_model2/LeNet5.h5'
    image_path='img/img.png'
    evaluate(model_path=model_path,image_path=image_path)










'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 6)         156       
_________________________________________________________________
average_pooling2d (AveragePo (None, 14, 14, 6)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416      
_________________________________________________________________
average_pooling2d_1 (Average (None, 5, 5, 16)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 1, 1, 120)         48120     
_________________________________________________________________
flatten (Flatten)            (None, 120)               0         
_________________________________________________________________
dense (Dense)                (None, 84)                10164     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                850       
=================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
_________________________________________________________________


Epoch 1/10
1875/1875 [==============================] - 9s 4ms/step - loss: 0.8786 - acc: 0.7064 - val_loss: 0.2428 - val_acc: 0.9303
Epoch 2/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.1866 - acc: 0.9428 - val_loss: 0.1408 - val_acc: 0.9578
Epoch 3/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1183 - acc: 0.9640 - val_loss: 0.0968 - val_acc: 0.9700
Epoch 4/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0882 - acc: 0.9731 - val_loss: 0.0781 - val_acc: 0.9751
Epoch 5/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0721 - acc: 0.9779 - val_loss: 0.0616 - val_acc: 0.9801
Epoch 6/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0596 - acc: 0.9817 - val_loss: 0.0539 - val_acc: 0.9823
Epoch 7/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0524 - acc: 0.9836 - val_loss: 0.0466 - val_acc: 0.9848
Epoch 8/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0457 - acc: 0.9858 - val_loss: 0.0504 - val_acc: 0.9834
Epoch 9/10
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0405 - acc: 0.9876 - val_loss: 0.0531 - val_acc: 0.9834
Epoch 10/10
1875/1875 [==============================] - 10s 6ms/step - loss: 0.0373 - acc: 0.9885 - val_loss: 0.0475 - val_acc: 0.9846
313/313 [==============================] - 1s 3ms/step - loss: 0.0475 - acc: 0.9846

'''