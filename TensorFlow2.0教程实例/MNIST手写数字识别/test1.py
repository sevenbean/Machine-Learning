import  tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
(train_images,train_lables),(test_images,test_lables)=mnist.load_data()
def plt_image(image):
    plt.imshow(image.reshape(28,28),cmap="binary")
    plt.show()
plt_image(train_images[2])