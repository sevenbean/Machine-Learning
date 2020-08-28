import urllib.request
import os
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt

# 第一步：下载数据
data_url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
data_file_path = "titanic3.xls"
if not os.path.isfile(data_file_path):
    result = urllib.request.urlretrieve(data_url, data_file_path)
    print("downloaded:", result)
else:
    print(data_file_path, "data file already exists")
# 读取数据
data = pd.read_excel("titanic3.xls")
selected_columns = ["survived", "name", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
select_data = data[selected_columns]


def prepare_data(select_data):
    # 筛选数据:对数据进行预处理：

    # 判断这些数据中含有空值的列名
    # print(select_data.isnull().any())
    # 处理数据：将数值中存在null时，将这一列的平均值填充进去
    # 将年龄中存在空白值的填充平均值进去：
    select_data = select_data.drop(["name"], axis=1)
    select_data["age"] = select_data["age"].fillna(select_data["age"].mean())
    # 将fare中空值中填充平均值：
    select_data["fare"] = select_data["fare"].fillna(select_data["fare"].mean())
    # 为记录embared填充空值：
    select_data["embarked"] = select_data["embarked"].fillna("S")
    # 将性别的值映射成0,1
    select_data["sex"] = select_data["sex"].map({"male": 1, "female": 0}).astype(int)
    # 将embarked映射成0,1,2
    select_data["embarked"] = select_data["embarked"].map({"C": 0, "Q": 1, "S": 2}).astype(int)
    # 由于在训练的时候不需要name这个字段，故可以把这个字段给删除，axis=1删除这一列

    ndarray = select_data.values
    # 后7列都是特征值：
    features = ndarray[:, 1:]
    # 标签
    labels = ndarray[:, 0]
    # 对那些特征值进行归一化处理：将值映射到0-1之间
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    normal_features = minmax_scale.fit_transform(features)
    return normal_features, labels


shuffle_df_data = select_data.sample(frac=1)
x_data, y_data = prepare_data(shuffle_df_data)

train_size = int(len(x_data) * 0.8)
x_train = x_data[:train_size]
y_train = y_data[:train_size]
x_test = x_data[train_size:]
y_test = y_data[train_size:]

print(x_train.shape)
print(y_train.shape)
#构建训练模型：
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=64, input_dim=7, use_bias=True, activation="relu"))
model.add(tf.keras.layers.Dense(units=32, activation="sigmoid"))
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="binary_crossentropy", metrics=["accuracy"])
train_history = model.fit(x_train, y_train, batch_size=40, validation_split=0.2, epochs=100, verbose=2)
test_history=model.evaluate(x_test,y_test)
# y=model.predict_classes(x_test)
# print("预测值：",y[3],"正确值：",y_test[3])
# Jack_info=[0,"jack",3,"male",23,1,0,5.000,"S"]
# Rose_info=[1,"Rose",1,"female",20,1,0,100.0000,"S"]
# new_passenger_pd=pd.DataFrame([Jack_info,Rose_info],columns=selected_columns)
# all_passenger=select_data.append(new_passenger_pd)
# shuffle_df_data = all_passenger.sample(frac=1)
# x_data, y_data = prepare_data(shuffle_df_data)
# y_predict=model.predict(x_data)
# all_passenger.insert(len(all_passenger.columns),"survived_properity",y_predict)
# print("预测结果：",all_passenger[-5:])
print(test_history)
# 数据可视化
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(["train","validation"],loc="upper left")
    plt.show()
show_train_history(train_history,"loss","val_loss")
show_train_history(train_history,"accuracy","val_accuracy")