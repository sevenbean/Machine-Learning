from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import os
import PIL.Image
import time
from IPython import display
import matplotlib.pyplot as plt
import glob
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5
#将标签进行独热编码。
train_labels=tf.one_hot(train_labels,depth=10)
train_labels=tf.cast(train_labels,tf.float32)
buffer_size = 60000
batch_size = 256
# 批量化和打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(buffer_size).batch(batch_size)


# 构建生成器模型
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=7 * 7 * 256, use_bias=False, input_shape=(110,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 28, 28, 1)
    return model


# 定义判别器：
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 11]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model


# 定义二元损失交叉熵函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# 定义判别器损失
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# 定义生成器损失函数
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 定义两者的优化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator = make_generator_model()
generator.summary()
discriminator = make_discriminator_model()
discriminator.summary()
# 保存检查点
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer, generator=generator,
                                 discriminator=discriminator)
# 定义超参数
train_epochs = 2
noise_dim = 100
num_examples_to_generator = 100
seed = tf.random.normal([num_examples_to_generator, noise_dim])
# 有规律的设置编码
labels=[i%10 for i in range(num_examples_to_generator)]
labels=tf.one_hot(labels,depth=10)
labels=tf.cast(labels,tf.float32)
seed=tf.concat([seed,labels],1)


# 定义单步的训练过程
@tf.function
def train_step(data_batch):
    images=data_batch[0]
    labels=data_batch[1]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        noise=tf.random.normal([images.get_shape()[0],noise_dim])
        noise_input=tf.concat([noise,labels],1)
        generator_images=generator(noise_input,training=True)
        labels_input=tf.reshape(labels,[images.get_shape()[0],1,1,10])
        images_input=tf.concat([images,labels_input*tf.ones([images.get_shape()[0],28,28,10])],3)
        generator_input=tf.concat([generator_images,labels_input*tf.ones([images.get_shape()[0],28,28,10])],3)

        real_output = discriminator(images_input, training=True)
        fake_output = discriminator(generator_input, training=True)
        gen_loss = generator_loss(fake_output)
        real_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(real_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# 定义保存图片
def generator_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig("C_image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


# 定义训练函数
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for data_batch in dataset:
            train_step(data_batch)
        display.clear_output(wait=True)
        generator_and_save_images(generator,epoch+1,seed)
        if (epoch+1)%5==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print("Time for epoch {} is {} sec".format(epoch+1,time.time()-start))

train(train_dataset, train_epochs)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def display_image(epoch_no):
    return PIL.Image.open("C_image_at_epoch_{:04d}.png".format(epoch_no))


display_image(1)
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)
