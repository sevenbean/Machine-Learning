from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np
import tensorflow as tf

from tFlow.图片识别.ImageShow import ImageShow

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# 数据进行标准化处理：
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
model = tf.keras.models.Sequential()
# 添加卷积1层
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation="relu", padding="same"))
# 防止过拟合
model.add(tf.keras.layers.Dropout(rate=0.3))
# 添加最大池化层
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
# 添加卷积2层
# model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
# # 防止过拟合
# model.add(tf.keras.layers.Dropout(rate=0.3))
# # 添加第二个最大池化层：
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
# 添加平坦层：
model.add(tf.keras.layers.Flatten())
# 添加输出层：
model.add(tf.keras.layers.Dense(10, activation="softmax"))

train_times = 10
batch_size = 100
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
train_history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=train_times, verbose=2,
                          validation_split=0.2)
test_history=model.evaluate(test_images,test_labels)
pred=model.predict_classes(test_images)
ig=ImageShow(test_images,test_labels,pred,0)
# ig.showImage()
ig.show_train_history(train_history,"loss","val_loss")
ig.show_train_history(train_history,"accuracy","val_accuracy")


