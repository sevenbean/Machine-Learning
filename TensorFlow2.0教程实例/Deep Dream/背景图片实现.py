import tensorflow as tf
import numpy as np
import PIL.Image
import time
# 图像数据处理
# 图像标准化：
def readImage(filename,max_dim=None):
    img=PIL.Image.open(filename)
    if max_dim:
        img.thumbnail((max_dim,max_dim))
    return  np.array(img)

def normalize_image(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)
# 展示图片
def show_image(img):
    PIL.Image.fromarray(np.array(img)).show()
# 保存图片
def save_image(img, file_name):
    PIL.Image.fromarray(np.array(img)).save(file_name)



#构建模型
base_model=tf.keras.applications.InceptionV3(include_top=False,weights="imagenet")
#最大限度的激活这些层的指定层
layer_names=["mixed3","mixed5"]
layers=[base_model.get_layer(name).output for name in layer_names]
#创建提取特征模型
dream_model=tf.keras.Model(inputs=base_model.input,outputs=layers)
# dream_model.summary()
#定义损失函数：
def calc_loss(img,model):
    # 选定第13通道
    #对图形进行变形，由（300，300，3）扩展为（1,300,300，3）
    img=tf.expand_dims(img,axis=0)
    #图像通过模型前向传播算法计算得到的结果
    layer_activations=model(img)
    #取选中通道的值
    losses=[]
    for i in layer_activations:
        loss=tf.math.reduce_mean(i)
        losses.append(loss)
    return tf.reduce_sum(losses)
# 定义优化过程函数
def render_deepdream(model,img,steps=100,step_size=0.01,verbose=1):
    for i in tf.range(steps):
        with tf.GradientTape() as tape:
            #对img进行梯度变换
            tape.watch(img)
            loss=calc_loss(img,model)
        #计算损失相对于输入图像像素的梯度
        gradients=tape.gradient(loss,img)
        # 归一化梯度值
        gradients/=tf.math.reduce_std(gradients)+1e-8
        # 在梯度上升中损失值越来越大，因此可以直接添加损失值到图像中
        img=img+gradients*step_size
        img=tf.clip_by_value(img,-1,1)
        #输出过程提示信息
        if verbose==1:
            if (i+1)%10==0:
                print("step {}/{},loss {}".format(i+1,steps,loss))
    return  img

original_img=readImage("news.jpg",max_dim=500)
img=tf.keras.applications.inception_v3.preprocess_input(original_img)
img=tf.convert_to_tensor(img)
start=time.time()
print("开始做梦")
dream_img=render_deepdream(dream_model,img,steps=100,step_size=0.01)
end=time.time()
print("梦醒时分：",end-start)
dream_img=normalize_image(dream_img)
show_image(dream_img)
filename="deepdream_{}背景图片.jpg".format(layer_names)
save_image(dream_img,filename)
print("梦境已经保存{}".format(filename))