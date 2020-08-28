import numpy as np
t1=np.arange(30).reshape(5,6).astype(np.float)

t1[[1,3]]=np.nan
print(t1)
for i in range(t1.shape[1]):
    temp_col=t1[:,i]
    num_isnan=np.count_nonzero(temp_col)
    if num_isnan >0:
        temp_col[temp_col!=temp_col]=temp_col[temp_col==temp_col].mean()
print(t1)