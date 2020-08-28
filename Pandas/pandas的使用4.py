import pandas as pd
df=pd.read_csv("E:\\ASUS\\data\\data\\boston.csv")
print("*"*20)
print(df.index)
print(df.columns)
print(df.values)
print(df.shape)
print(df.info())
print(df.describe())
print(df.max())
print(df.mean())
df=df.sort_values(by="CRIM",ascending=False)

print(df.head())