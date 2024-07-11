import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


def BasicBlock(filter_num, strides, _inputs):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=3, strides=strides, padding='same')(_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filter_num, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides != 1:
        y = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=1, strides=strides)(_inputs)
        y = tf.keras.layers.BatchNormalization()(y)
    else:
        y = _inputs

    output = tf.keras.layers.add([x, y])
    output = tf.keras.layers.Activation('relu')(output)

    return output



def ResNet18():  # 2 2 2 2
    input_image = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(input_image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = BasicBlock(64, strides=1, _inputs=x)
    x = BasicBlock(64, strides=1, _inputs=x)

    x = BasicBlock(128, strides=2, _inputs=x)
    x = BasicBlock(128, strides=1, _inputs=x)

    x = BasicBlock(256, strides=2, _inputs=x)
    x = BasicBlock(256, strides=1, _inputs=x)

    x = BasicBlock(512, strides=2, _inputs=x)
    x = BasicBlock(512, strides=1, _inputs=x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_image, outputs=x)
    return model



def BottleNeck(filter_num, strides, _inputs, down=False):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=1, strides=1, padding='same')(_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=3, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides != 1 or down == True:
        y = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=1, strides=strides)(_inputs)
        y = tf.keras.layers.BatchNormalization()(y)
    else:
        y = _inputs

    output = tf.keras.layers.add([x, y])
    output = tf.keras.layers.Activation('relu')(output)

    return output


'''模型定义
'''
def ResNet50():  # 3 4 6 3
    input_image = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(input_image)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    x = BottleNeck(filter_num=64, strides=1, _inputs=x, down=True)
    x = BottleNeck(filter_num=64, strides=1, _inputs=x)
    x = BottleNeck(filter_num=64, strides=1, _inputs=x)

    x = BottleNeck(filter_num=128, strides=2, _inputs=x)
    x = BottleNeck(filter_num=128, strides=1, _inputs=x)
    x = BottleNeck(filter_num=128, strides=1, _inputs=x)
    x = BottleNeck(filter_num=128, strides=1, _inputs=x)

    x = BottleNeck(filter_num=256, strides=2, _inputs=x)
    x = BottleNeck(filter_num=256, strides=1, _inputs=x)
    x = BottleNeck(filter_num=256, strides=1, _inputs=x)
    x = BottleNeck(filter_num=256, strides=1, _inputs=x)
    x = BottleNeck(filter_num=256, strides=1, _inputs=x)
    x = BottleNeck(filter_num=256, strides=1, _inputs=x)

    x = BottleNeck(filter_num=512, strides=2, _inputs=x)
    x = BottleNeck(filter_num=512, strides=1, _inputs=x)
    x = BottleNeck(filter_num=512, strides=1, _inputs=x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_image, outputs=x)
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
    # model = ResNet50()
    model = ResNet18()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['acc'])


    '''训练
    训练epoch轮
    '''
    history = model.fit(train_gen, epochs=10, validation_data=val_gen)


    '''训练完毕进行评估
    可以根据各种指标计算模型性能，并保存模型
    '''
    plt.plot(history.epoch, history.history.get('acc'))
    plt.plot(history.epoch, history.history.get('val_acc'))
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



if __name__ == "__main__":
    # GPU内存管理
    gpus = tf.config.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    data_path = "sat/"
    model_path = 'saved_model/my_model2/ResNet.h5'
    train(data_path=data_path, model_path=model_path)

    model_path = 'saved_model/my_model2/ResNet.h5'
    image_path = 'img/img.png'
    evaluate(model_path=model_path, data_path=image_path)