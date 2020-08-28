import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
valid_split = 0.2
train_num = int(len(train_images) * (1 - valid_split))

# 第一步：准备数据
# 训练数据集
train_x = train_images[:train_num]
train_y = train_labels[:train_num]
# 验证数据集
valid_x = train_images[train_num:]
valid_y = train_labels[train_num:]
# 测试数据集
test_x = test_images
test_y = test_labels
# 将值拉成一位数组
train_x.shape = (-1, 784)
valid_x.shape = (-1, 784)
test_x.shape = (-1, 784)
# 数据归一化
train_x = tf.cast(train_x / 255.0, tf.float32)
valid_x = tf.cast(valid_x / 255.0, tf.float32)
test_x = tf.cast(test_x / 255.0, tf.float32)
# 独热编号
train_y = tf.one_hot(train_y, depth=10)
valid_y = tf.one_hot(valid_y, depth=10)
test_y = tf.one_hot(test_y, depth=10)
# 定义超参数
w = tf.Variable(tf.random.normal([28 * 28, 10], mean=0.0, stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.zeros([10]), tf.float32)


# 定义模型
def model(x, w, b):
    pred = tf.matmul(x, w) + b
    return tf.nn.sigmoid(pred)


# 第一种定义交叉熵损失函数
def loss_function(x, y, w, b):
    pred = model(x, w, b)
    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=pred)
    return tf.reduce_mean(loss_)




# 定义梯度计算函数
def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss_function(x, y, w, b)
    return tape.gradient(loss_, [w, b])


# 定义准确率：
def accurate(w, x, b, y):
    pred = model(x, w, b)
    correct_predict = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# 定义优化器
train_times = 20
learning_rate = 0.01
batch_step = 50
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

step = int(train_num / batch_step)
loss_train_list = []
loss_vaild_list = []
acc_train_list = []
acc_valid_list = []
for i in range(train_times):
    for r in range(step):
        xs = train_x[r * batch_step:(r + 1) * batch_step]
        ys = train_y[r * batch_step:(r + 1) * batch_step]
        grads = grad(xs, ys, w, b)
        optimizer.apply_gradients(zip(grads, [w, b]))
    loss_train = loss_function(train_x, train_y, w, b).numpy()
    loss_valid = loss_function(valid_x, valid_y, w, b).numpy()
    acc_train = accurate(w, train_x, b, train_y).numpy()
    acc_valid = accurate(w, valid_x, b, valid_y).numpy()
    loss_train_list.append(loss_train)
    loss_vaild_list.append(loss_valid)
    acc_train_list.append(acc_train)
    acc_valid_list.append(acc_valid)
    # print("w的值：", w.numpy(), "b的值：", b.numpy(), "训练正确率：", acc_train, "验证的正确率", acc_valid)
# plt.xlabel("训练")
# plt.ylabel("boss")
# plt.plot(loss_train_list,"blue",label="Train_boss")
# plt.plot(loss_vaild_list,"red",label="Valid_loss")
# plt.show()
# 获得测试集的正确率：
acc = accurate(w, test_x, b, test_y)
print(acc.numpy())


# 应用模型：退出测试值
# print("预测值：",model(test_x[1],w,b),"正确值：",test_y[1])
def predict(w, x, b):
    pred = model(x, w, b)
    result = tf.argmax(pred, 1).numpy()
    return result


pred = predict(w, test_x, b)
print(pred[1], tf.argmax(test_y[1]))


def picture(images, labels, pred, index, num=10):
    fig = plt.gcf()  # 获取当前图表
    fig.set_size_inches(10, 12)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(np.reshape(images[index], (28, 28)), cmap="binary")
        title = "label=" + str(np.argmax(labels[index]))
        if len(pred) > 0:
            title = "pred=" + str(pred[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()


picture(test_x, test_y, pred, 1, 25)
