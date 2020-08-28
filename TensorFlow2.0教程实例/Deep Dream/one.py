import numpy as np
import PIL.Image as Image
import  tensorflow as tf
image_string=Image.open("news.jpg","r")
image_string=np.array(image_string)
print(image_string)
