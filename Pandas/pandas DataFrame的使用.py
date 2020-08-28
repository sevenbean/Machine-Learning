import pandas as pd

# 使用字典构建DataFrame
student_dict = {"name": ["张三", "李四", "王五"], "age": [25, 23, 12], "sex": ["male", "female", "male"]}
pd1 = pd.DataFrame(student_dict)
print(pd1)
# 使用列表来构建DataFrame
student_list = [{"name": "张三", "age": 15, "sex": "male"}, {"name": "李四", "age": 30, "sex": "male"},
                {"name": "王五", "age": 23, "sex": "female"}]
pd2=pd.DataFrame(student_list)
print(pd2)
