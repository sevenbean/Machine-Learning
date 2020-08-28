import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#第一步：准备数据
x_data = np.linspace(-1, 1, 100)
y_data = 2*x_data + np.random.randn(*x_data.shape) * 0.4
# 第二步：创建模型
def model(w,x,b):
    return w*x+b
#第三步：定义损失函数
def loss_function(w,x,y,b):
    loss=y-model(w,x,b)
    return tf.reduce_mean(tf.square(loss))
#第四步：定义梯度函数：
def grad(w,x,y,b):
    with tf.GradientTape() as tape:
        loss=loss_function(w,x,y,b)
    return tape.gradient(loss,[w,b])
w=tf.Variable(np.random.randn(),tf.float32,name="W")
b=tf.Variable(1.0,tf.float32,name="b")
train_time=10
learning_rate=0.01
loss_list=[]
optimizer=tf.keras.optimizers.SGD(learning_rate)
for i in range(train_time):
    for xs,ys in zip(x_data,y_data):
        grads=grad(w,xs,ys,b)
        optimizer.apply_gradients(zip(grads,[w,b]))
    loss_=loss_function(w,xs,ys,b).numpy()
    print("w的值", w.numpy(), "b的值",b.numpy(), "平均损失：",loss_)
    loss_list.append(loss_)

plt.scatter(x_data,y_data,color="green")
plt.plot(x_data,w.numpy()*x_data+b.numpy())
plt.show()
# plt.scatter(x_data, y_data)
# plt.show()
