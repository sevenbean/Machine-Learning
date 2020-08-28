import tensorflow as tf
import numpy as np
import pandas as pd
from  sklearn.utils import  shuffle
from sklearn.preprocessing import scale
#第一步：准备数据
df=pd.read_csv("E:\\ASUS\\data\\data\\boston.csv")
df=np.array(df.values)

x_data=df[:,:12]
y_data=df[:,12]
#训练集
train_num=300
train_xdata=x_data[:train_num]
train_xdata=tf.cast(scale(train_xdata),tf.float32)
train_ydata=y_data[:train_num]
#验证集
valid_num=100
valid_xdata=x_data[train_num:train_num+valid_num]
#使用scale函数可以进行数据的归一化操作
valid_xdata=tf.cast(scale(valid_xdata),tf.float32)
valid_ydata=y_data[train_num:train_num+valid_num]
#测试集
test_num=len(y_data)-train_num-valid_num
test_xdata=x_data[valid_num+train_num:]
test_ydata=y_data[valid_num+train_num:]
#定义模型
def model(w,x,b):
    return tf.matmul(x,w)+b
# 定义一些变量
w=tf.Variable(tf.random.normal(shape=(12,1),mean=0.0,stddev=1.0))#mean是平均值，stddev表示的标准差
b=tf.Variable(tf.zeros(1),tf.float32)
#定义损失函数
def loss_function(x,y,w,b):
    err=model(w,x,b)-y
    loss=tf.square(err)
    return tf.reduce_mean(loss)
#定义一个梯度计算函数
def grad(x,y,w,b):
    with tf.GradientTape() as tp:
        loss_=loss_function(x,y,w,b)
    return tp.gradient(loss_,[w,b])
train_time=50
learning_rate=0.001
batch_size=10
#定义一个优化器
optimizer=tf.keras.optimizers.SGD(learning_rate)#使用梯度下降算法
loss_list_train=[]
loss_list_valid=[]
total_step=int(train_num/batch_size)
for i in range(train_time):
    for step in range(total_step):
        xs=train_xdata[step*batch_size:(step+1)*batch_size,:]
        ys=train_ydata[step*batch_size:(step+1)*batch_size]
        #计算梯度
        grads=grad(xs,ys,w,b)
        #优化器根据梯度自动调整W和B的值
        optimizer.apply_gradients(zip(grads,[w,b]))
    loss_train=loss_function(train_xdata,train_ydata,w,b).numpy()
    loss_valid=loss_function(valid_xdata,valid_ydata,w,b).numpy()
    loss_list_valid.append(loss_valid)
    loss_list_train.append(loss_train)
    print("w的值:",w,"b的值",b.numpy(),"第",(i+1),"次","loss_train:",loss_train,"loss_valid:",loss_valid)

