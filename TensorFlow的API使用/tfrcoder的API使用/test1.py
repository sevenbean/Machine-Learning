#tfrecoder文件格式的使用
import tensorflow as tf
favo_books=[name.encode("utf8") for name in ["machine learning","cc150"]]
favo_books_bytelist=tf.train.BytesList(value=favo_books)
print(favo_books_bytelist)
favo_books_Floatlist=tf.train.FloatList(value=favo_books)
print(favo_books_Floatlist)
