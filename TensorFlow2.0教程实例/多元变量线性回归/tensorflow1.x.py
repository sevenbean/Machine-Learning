import  tensorflow.compat.v1 as tf
import  numpy as np
import pandas as pd
from sklearn.utils import  shuffle
import matplotlib.pyplot as plt
tf.disable_eager_execution()
df=pd.read_csv("E:\\ASUS\\data\\data\\boston.csv")
df=np.array(df.values)
# 对于一些值偏差非常大的数据进行归一化处理
for i in range(12):
    df[:,i]=df[:,i]/(df[:,i].max()-df[:,i].min())
# 准备好数据
x_data=df[:,:12]
y_data=df[:,12]


# 创建两个placeholder
x=tf.placeholder(tf.float32,[None,12],name="X")
y=tf.placeholder(tf.float32,[None,1],name="y")

w=tf.Variable(tf.random_normal([12,1],stddev=0.01),name="w")
b=tf.Variable(1.0)
# 创建模型
with tf.name_scope("Model"):
    def model(w,x,b):
        return tf.matmul(x,w)+b
    pred=model(w,x,b)
#定义损失函数
loss_function=tf.reduce_mean(tf.square(y-pred))
# 创建模型
train_times=50
learning_rate=0.01
#定义优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
init=tf.global_variables_initializer()
loss_list=[]
# 开始训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_times):
        loss_sum=0
        for xs,ys in zip(x_data,y_data):
            xs=xs.reshape(1,12)
            ys=ys.reshape(1,1)
            optm,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
            loss_list.append(loss)
            loss_sum+=loss
        shuffle(x_data,y_data)
        loss_avg=loss_sum/len(y_data)
    b0temp = b.eval()
    w0temp = w.eval()
        # print("损失的平均值=",loss_avg,"b=",b0temp,"w=",w0temp)
# 进行预测
with tf.Session() as sess:
    print("b的值:",b0temp)
    test_position=np.random.randint(506)
    print("下标：",test_position)
    test_x=x_data[test_position]
    print("预测值：",sess.run(tf.matmul(tf.cast(test_x.reshape(1,12),tf.float32),w0temp)+b0temp))
    # print("预测值：",sess.run(model(w0temp,tf.cast(test_x.reshape(1,12),tf.float32),b0temp)))
    print("真的值：",df[test_position,12])
    plt.plot(loss_list)
