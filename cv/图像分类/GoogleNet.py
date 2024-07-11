import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Inception(ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, input_):
    inputs = tf.keras.layers.Input(shape=input_.shape[1:])
    x1 = tf.keras.layers.Conv2D(ch1x1, kernel_size=1, activation="relu")(inputs)

    x21 = tf.keras.layers.Conv2D(ch3x3red, kernel_size=1, activation="relu")(inputs)
    x22 = tf.keras.layers.Conv2D(ch3x3, kernel_size=3, padding="same", activation="relu")(x21)

    x31 = tf.keras.layers.Conv2D(ch5x5red, kernel_size=1, activation="relu")(inputs)
    x32 = tf.keras.layers.Conv2D(ch5x5, kernel_size=5, padding="same", activation="relu")(x31)

    x41 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inputs)
    x42 = tf.keras.layers.Conv2D(pool_proj, kernel_size=1, activation="relu")(x41)
    outputs = tf.concat((x1, x22, x32, x42), axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def InceptionAux(num_classes, input_):
    inputs = tf.keras.layers.Input(shape=input_.shape[1:])
    x = tf.keras.layers.AvgPool2D(pool_size=5, strides=3)(inputs)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate=0.7)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.7)(x)
    x = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

'''模型定义
'''
def GoogLeNet():
    input_image = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same", activation="relu")(input_image)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    # 3a
    x = Inception(64, 96, 128, 16, 32, 32, x)(x)
    # 3b
    x = Inception(128, 128, 192, 32, 96, 64, x)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    # 4a
    x = Inception(192, 96, 208, 16, 48, 64, x)(x)
    # aux1
    aux11 = InceptionAux(5, x)(x)
    aux1 = tf.keras.layers.Softmax(name="aux_1")(aux11)
    # 4b
    x = Inception(160, 112, 224, 24, 64, 64, x)(x)
    # 4c
    x = Inception(128, 128, 256, 24, 64, 64, x)(x)
    # 4d
    x = Inception(112, 144, 288, 32, 64, 64, x)(x)
    # aux2
    aux22 = InceptionAux(5, x)(x)
    aux2 = tf.keras.layers.Softmax(name="aux_2")(aux22)
    # 4e
    x = Inception(256, 160, 320, 32, 128, 128, x)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    # 5a
    x = Inception(256, 160, 320, 32, 128, 128, x)(x)
    # 5b
    x = Inception(384, 192, 384, 48, 128, 128, x)(x)
    x = tf.keras.layers.AvgPool2D(pool_size=7, strides=1)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(5)(x)
    # aux3
    aux3 = tf.keras.layers.Softmax(name="aux_3")(x)

    model = tf.keras.models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    #     model = tf.keras.models.Model(inputs=input_image, outputs=aux3)
    return model



def train(data_path, model_path):
    '''数据处理
        目标：原数据->模型能读入的数据(分为训练集和测试集，细分为输入数据和输出结果，再细分为batch)
    '''
    train_dir = data_path.join('/train')
    test_dir = data_path.join('/val')
    im_size = 224
    batch_size = 32
    train_images = ImageDataGenerator(rescale=1 / 255, horizontal_flip=True)
    test_images = ImageDataGenerator(rescale=1 / 255)
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

    '''模型导入
    导入模型，设置优化器，损失函数
    '''
    model = GoogLeNet()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['acc'])


    '''训练
    训练epoch轮
    '''
    history = model.fit(train_gen, epochs=20, validation_data=val_gen)


    '''训练完毕进行评估
    可以根据各种指标计算模型性能，并保存模型
    '''
    plt.plot(history.epoch, history.history.get('aux_3_acc'))
    plt.plot(history.epoch, history.history.get('val_aux_3_acc'))
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
    model_path = 'saved_model/my_model2/GoogleNet.h5'
    train(data_path=data_path, model_path=model_path)

    model_path = 'saved_model/my_model2/GoogleNet.h5'
    image_path = 'img/img.png'
    evaluate(model_path=model_path, data_path=image_path)



