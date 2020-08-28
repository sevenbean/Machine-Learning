import numpy as np

t1=np.arange(30).reshape(5,6)
print(t1)
print("-"*20)
print(t1[:,2])
print("-"*20)
print(t1[1,:])
print("-"*20)
print(t1[1:5,1:3])
print("-"*20)
print(t1[1:, 2:])
#选取其中的多个不连续的行
print(t1[[1, 3],:])
#选取其中多个不连续的列
print("-"*20)
print(t1[:, [2, 4]])
print("-"*20)
print(t1)
print("-"*20)
print(t1[:, [1, 3, 4]])
print("高级索引------boolean索引")
b=np.random.randint(0,30,(5,6))
print(b)
print(b>15)

print("-"*20)
#将数组中大于15的值的赋值为0
# b[b>15]=0
# print(b)
#将小于10的赋值为10，将大于10的赋值为20
# print(b)
# print(np.where(b<10,10,20))

# 将小于10的赋值为20，大于20的赋值为20
print(b.clip(10, 20))