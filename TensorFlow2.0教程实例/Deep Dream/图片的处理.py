import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
#图片的编码处理
image_raw_data=tf.gfile.FastGFile("caiweixin.png","rb").read()
with tf.Session() as sess:
    image_data=tf.image.decode_png(image_raw_data)
    #图像的缩放：由于网络上的图片大小是不固定的，所以选择进行图片大小的设置
    #使用不同的缩放方法只需要修改method=0，1,2,3
    # resized1=tf.image.resize_images(image_data,[256,256],method=1)
    # resized1=np.asarray(resized1.eval(),dtype="uint8")
    #图像的裁剪:
    #图像裁剪1：如果目标图像的尺寸小于原始图像的尺寸，则在中心位置裁剪，反之则用黑色的像素进行填充
    croped=tf.image.resize_image_with_crop_or_pad(image_data,400,400)
    #图像裁剪2：随机裁剪
    croped_random=tf.image.random_crop(image_data,[200,200,3])
    #图像的翻转
    filt_left=tf.image.flip_left_right(image_data)
    filt_up_down=tf.image.flip_up_down(image_data)
    # print(image_data.eval())
    # plt.imshow(croped.eval())
    # plt.imshow(croped_random.eval())
    # plt.imshow(filt_left.eval())
    plt.imshow(filt_up_down.eval())
    plt.show()
