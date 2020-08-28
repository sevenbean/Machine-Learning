# 使用kernels创建多层神经元
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 对数据进行归一化处理
test_images = test_images / 255.0
train_images = train_images / 255.0
# 第一种：对标签设置成独热编码
# train_labels_oht = tf.one_hot(train_labels, depth=10)
# test_labels_oht = tf.one_hot(test_labels, depth=10)

# 创建一个keras空的序列
model = tf.keras.models.Sequential()
# 添加输入层(平坦层，Flatten)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 添加隐藏层(密基层，Dense)units:表示这一层神经元个数，kernel_inirializer表示权重，activation表示激活函数
model.add(tf.keras.layers.Dense(units=64, kernel_initializer="normal", activation="relu"))
model.add(tf.keras.layers.Dense(units=32, kernel_initializer="normal", activation="relu"))
# 添加输出层：h
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
# # 输出模型摘要：
# model.summary()
# 定义训练模型：optimizer表示的是优化器，loss表示的损失函数，metrics表示评估的价值
#采用第二种：不需要设置标签的独热编码，sparse_categorical_crossentropy会自动的对标签值进行独热编码
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
train_time = 10
batch_size = 50
# 训练模型：
train_history = model.fit(train_images, train_labels, validation_split=0.2, epochs=train_time,
                          batch_size=batch_size, verbose=2)
#测试模型：
test_history=model.evaluate(test_images,test_labels,verbose=2)
#应用模型：
test_pred=model.predict_classes(test_images)
print(test_pred[0])
tf.one_hot(test_labels,depth=10)
print(test_labels[0])
# print(test_history)
# print(train_history)
#数据可视化：
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(["train","validation"],loc="upper left")
    plt.show()
show_train_history(train_history,"loss","val_loss")
show_train_history(train_history,"accuracy","val_accuracy")

