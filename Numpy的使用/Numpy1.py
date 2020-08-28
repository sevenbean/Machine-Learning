import numpy as np
t1=np.array(np.arange(12,24)).reshape(3,4)
t2=np.array(np.arange(0,12)).reshape(3,4)
t1_zeros=np.zeros((t1.shape[0],1),dtype=np.int)
t2_ones=np.ones((t2.shape[0],1),dtype=np.int)
print(t1_zeros)
print(t2_ones)
print(np.hstack((t1, t1_zeros)))
print(np.hstack((t2, t2_ones)))
# print(t1)
# print(t2)
#垂直连接
# print(t1.shape)
# print(t2.shape)
# print(np.vstack((t1, t2)))
# #水平连接
# print(np.hstack((t1, t2)))
# # 行的交换
# print("*"*20)
# t1[[1,2],:]=t1[[2,1],:]
# print(t1)
# #列的交换
# t2[:,[1,2]]=t2[:,[2,1]]
# print(t2)