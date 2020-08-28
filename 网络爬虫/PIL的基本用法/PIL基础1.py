from PIL import  Image
img=Image.open("news.jpg")
img.show()
# 创建一个新的照片
# newImg=Image.new("RGBA",(640,250),(255,180,110))
# newImg.show()
# 点操作
# out=newImg.point(lambda i:i*2)
#通道分离
r,g,b=img.split()
print(r)