import tensorflow as tf
import numpy as np
import PIL.Image
import time


# 图像数据处理

def readImage(filename, max_dim=None):
    img = PIL.Image.open(filename)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# 定义随机移动图像
def random_roll(img, maxroll=512):
    shift = tf.random.uniform(shape=[2], minval=-maxroll,maxval=maxroll, dtype=tf.int32)
    print(shift)
    shift_down, shift_right = shift[0], shift[1]
    print(shift_down, shift_right)
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled


# 图像标准化：
def normalize_image(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# 展示图片
def show_image(img):
    PIL.Image.fromarray(np.array(img)).show()


# 保存图片
def save_image(img, file_name):
    PIL.Image.fromarray(np.array(img)).save(file_name)


# 构建模型
base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
# 最大限度的激活这些层的指定层
layer_names = ["mixed3", "mixed5"]
layers = [base_model.get_layer(name).output for name in layer_names]
# 创建提取特征模型
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


# dream_model.summary()
# 定义损失函数：
def calc_loss(img, model):
    # 选定第13通道
    # 对图形进行变形，由（300，300，3）扩展为（1,300,300，3）
    img = tf.expand_dims(img, axis=0)
    # 图像通过模型前向传播算法计算得到的结果
    layer_activations = model(img)
    # 取选中通道的值
    losses = []
    for i in layer_activations:
        loss = tf.math.reduce_mean(i)
        losses.append(loss)
    return tf.reduce_sum(losses)


# 求梯度
def get_tiled_gradients(model, img, title_size=150):
    shift_down, shift_right, img_rolled = random_roll(img, title_size)
    gradients = tf.zeros_like(img_rolled)
    xs = tf.range(0, img_rolled.shape[0], title_size)
    ys = tf.range(0, img_rolled.shape[1], title_size)
    for x in xs:
        for y in ys:
            # 计算该图块的梯度
            with tf.GradientTape() as tape:
                tape.watch(img_rolled)
                # 从图像中提取该图块，最后一块会按实际提取
                img_title = img_rolled[x:x + title_size, y:y + title_size]
                loss = calc_loss(img_title, model)
                gradients = gradients + tape.gradient(loss, img_rolled)
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    return gradients


def render_deepdream_with_octaves(model, img, steps_per_octave=100, step_size=0.01, octaves=range(-2, 3),
                                  octave_scale=1.3):
    inital_shape = img.shape[:-1]
    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(inital_shape), tf.float32) * (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))
        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(model, img)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
            if (step + 1) % 10 == 0:
                print("octave{},step{}".format(octave, step + 1))
        img = tf.image.resize(img, inital_shape)
        return normalize_image(img)

start = time.time()
original_img = readImage("news.jpg", max_dim=500)
OCT_SCALE = 1.30
img = tf.keras.applications.inception_v3.preprocess_input(original_img)
img = tf.convert_to_tensor(img)

img = render_deepdream_with_octaves(dream_model, img, steps_per_octave=50, step_size=0.01, octaves=range(-2, 3),
                                    octave_scale=1.3)


print("开始做梦")

show_image(img)
end = time.time()
print("梦醒时分：", end - start)
filename = "deepdream_{}背景图片优化2.jpg".format(layer_names)
save_image(img, filename)
print("梦境已经保存{}".format(filename))
