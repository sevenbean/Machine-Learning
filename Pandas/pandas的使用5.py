import pandas as pd
import numpy as np

pd1=pd.DataFrame(np.random.randint(0,30,(5,5)),index=list("ABCDE"),columns=list("VWXYZ"))
# 访问里面某个位置的值：
print(pd1.loc["A","W"])
print(pd1.loc["A",["W","X"]])
print(pd1.loc[["A", "C"], ["W", "Z"]])
print(type(pd1.loc["A",["W","Z"]]))
print(pd1)
print(pd1.loc["A":"C","W":"Z"])
print("***************iloc的使用*******************")
print(pd1.iloc[1,2])
print(pd1)
print(pd1.iloc[[1, 3], [2, 4]])
print(pd1.iloc[:,:2])
print("****************join连接***************")
df1=pd.DataFrame(np.random.randint(0,30,(2,4)),index=list("AB"),columns=list("WXYZ"))
df2=pd.DataFrame(np.random.randint(30,50,(3,5)),index=list("ABC"),columns=list("MNOPQ"))
print(df2.join(df1))
print(df1.merge(df1))
print()
