import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def train(data_path, model_path):
    '''数据处理
        目标：原数据->模型能读入的数据(分为训练集和测试集，细分为输入数据和输出结果，再细分为batch)
    '''
    train_dir = data_path.join('/train')
    test_dir = data_path.join('/val')
    im_size = 224
    batch_size = 32
    train_images = ImageDataGenerator(rescale = 1/255,horizontal_flip=True)
    test_images = ImageDataGenerator(rescale = 1/255)
    train_gen = train_images.flow_from_directory(directory=train_dir,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 target_size=(im_size, im_size),
                                                 class_mode='categorical')
    val_gen = test_images.flow_from_directory(directory=test_dir,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              target_size=(im_size, im_size),
                                              class_mode='categorical')
    classes = train_gen.class_indices




    '''模型定义
    '''
    model = tf.keras.Sequential()
    #VGG-11/16
    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 64,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    # model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
    #                                  filters = 64,
    #                                  kernel_size = (3,3),
    #                                  padding = 'same',
    #                                  activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 128,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    # model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
    #                                  filters = 128,
    #                                  kernel_size = (3,3),
    #                                  padding = 'same',
    #                                  activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 256,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 256,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    # model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
    #                                  filters = 256,
    #                                  kernel_size = (1,1),
    #                                  padding = 'same',
    #                                  activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 512,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 512,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    # model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
    #                                  filters = 512,
    #                                  kernel_size = (1,1),
    #                                  padding = 'same',
    #                                  activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 512,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
                                     filters = 512,
                                     kernel_size = (3,3),
                                     padding = 'same',
                                     activation = "relu"))
    # model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),
    #                                  filters = 512,
    #                                  kernel_size = (1,1),
    #                                  padding = 'same',
    #                                  activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    #model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    #model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.summary()
    '''或者这样定义
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vgg.trainable = False
    # 迁移学习 去掉全连接层 加载权重 输入尺寸
    model = tf.keras.Sequential()
    # VGG-11/16
    model.add(vgg)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.summary()
    '''




    '''模型导入
    导入模型，设置优化器，损失函数
    '''
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    '''训练
    训练epoch轮
    '''
    history = model.fit(train_gen,epochs=10,validation_data=val_gen)


    '''训练完毕进行评估
    可以根据各种指标计算模型性能，并保存模型
    '''
    plt.plot(history.epoch,history.history.get('acc'))
    plt.plot(history.epoch,history.history.get('val_acc'))
    model.evaluate(val_gen)
    model.save(model_path)



def evaluate(model_path,data_path):
    '''导入模型
    '''
    model = tf.keras.models.load_model(model_path)

    '''读取数据，转化成模型可以输入的格式
    '''
    img = cv2.imread(data_path, 1)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    img = img / 255

    '''调用模型得到结果
    '''
    predict = model.predict(img)


    '''对结果做处理，得到想要的数据
    '''
    label = ['airplane', 'bridge', 'palace', 'ship', 'stadium']
    print(label[np.argmax(predict)])


if __name__=="__main__":
    #GPU内存管理
    gpus = tf.config.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    data_path = "sat/"
    model_path = 'saved_model/my_model2/VGG_net.h5'
    train(data_path=data_path, model_path=model_path)

    model_path = 'saved_model/my_model2/VGG_net.h5'
    image_path = 'img/img.png'
    evaluate(model_path=model_path, data_path=image_path)