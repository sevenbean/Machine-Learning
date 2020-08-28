import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import scale

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
valid_split = 0.2
train_num = int(len(train_images) * (1 - valid_split))
# 准备好数据
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
# 确立中间神经元个数
Hnn = 64
Input_DIM=784
Output_Dim=10
# 权值和偏移量
W1 = tf.Variable(tf.random.normal([Input_DIM, Hnn], mean=0.0, stddev=1.0, dtype=tf.float32))
B1 = tf.Variable(tf.zeros([Hnn]), dtype=tf.float32)
W2 = tf.Variable(tf.random.normal([Hnn, Output_Dim], mean=0.0, stddev=1.0, dtype=tf.float32))
B2 = tf.Variable(tf.zeros([Output_Dim]), dtype=tf.float32)
# 创建权值列表和偏移量
W = [W1, W2]
B = [B1, B2]
# 创建模型
def model(x, w, b):
    x = tf.matmul(x, w[0]) + b[0]
    x = tf.nn.relu(x)
    x = tf.matmul(x, w[1]) + b[1]
    pred=tf.nn.softmax(x)
    return pred
# 定义交叉熵的损失函数：
def loss_function(x, y, w, b):
    pred = model(x, w, b)
    loss_=tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=pred)
    return tf.reduce_mean(loss_)


# 定义准确率：
def accuracy(x, y, w, b):
    pred = model(x, w, b)
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 定义梯度函数
def grad(x, y, w, b):
    var_list=w+b
    with tf.GradientTape() as tp:
        loss_ = loss_function(x, y, w, b)
    return tp.gradient(loss_,var_list)

train_epochs = 50
batch_size = 50
total_batch = int(train_num / batch_size)
learning_rate = 0.01
loss_train_list = []
loss_vaild_list = []
acc_train_list = []
acc_valid_list = []
# 选择优化器：
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for i in range(train_epochs):
    for r in range(total_batch):
        xs = train_x[r * batch_size:(r + 1) * batch_size]
        ys = train_y[r * batch_size:(r + 1) * batch_size]
        grads = grad(xs, ys, W, B)
        optimizer.apply_gradients(zip(grads,W+B))

    loss_train = loss_function(train_x, train_y, W, B).numpy()
    loss_vaild = loss_function(valid_x, valid_y, W, B).numpy()
    accu_train = accuracy(train_x, train_y, W, B).numpy()
    accu_valid = accuracy(valid_x, valid_y, W, B).numpy()
    loss_vaild_list.append(loss_vaild)
    loss_train_list.append(loss_train)
    acc_train_list.append(accu_train)
    acc_valid_list.append(accu_valid)
    print("训练的正确率：", accu_train, "验证的正确率：", accu_valid, "loss_train:", loss_train, "loss_valid:", loss_vaild)
