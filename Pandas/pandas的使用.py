import pandas as pd
import  numpy as np
ps=pd.Series(np.arange(10),index=list("abcdefghij"))
ps2=pd.Series(ps,index=list("fghij"))
print(ps2)
#通过说索引值来搜索
print(ps["j"])
print(ps[[1,3]])

print(ps.index)
print(ps.values)
ps=ps.astype(np.float)

print(ps)
# print(ps[[1,5,6]])
print(ps[["f","g","h"]])