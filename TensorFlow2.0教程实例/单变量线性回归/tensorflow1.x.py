import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pylab as plt

tf.disable_eager_execution()
# 第一步准备数据
x_data = np.linspace(-1, 1, 100)
y_data = 2 * x_data + 1 + np.random.randn(*x_data.shape) * 0.4


# 第二步：定义模型
def model(w, x, b):
    return w * x + b


w = tf.Variable(np.random.normal(), tf.float32)
b = tf.Variable(1.0)
x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")
pred = model(w, x, b)
# 第三步:定义损失函数
loss_function = tf.reduce_mean(tf.square(y - pred))
# 第四步定义一个优化器
train_time = 10
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
loss_list = []
# 第五步进行训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_time):
        loss_sum=0
        for xs, ys in zip(x_data, y_data):
            opt, loss = sess.run([optimizer, loss_function], feed_dict={x: xs,y:ys})
            loss_sum+=loss
            loss_list.append(loss)
        b0temp = b.eval(session=sess)
        w0temp = w.eval(session=sess)
        loss_avg=loss_sum/len(x_data)
        print("w的值",w0temp,"b的值",b0temp,"平均损失：",loss_avg)
# plt.scatter(x_data,y_data,color="red")
# plt.plot(x_data,x_data*w0temp+b0temp,color="blue")
plt.plot(loss_list)
plt.show()