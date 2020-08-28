class Cat:
    # 初始化方法
    def __init__(self,name,age):
        self.name=name
        self.age=age
    #销毁对象之后，运行该方法
    def __del__(self):
        print("%s轻轻的我走了"%(self.name))
    #修改对象默认输出的字符串
    def __str__(self):
        return "%s是大帅哥"%(self.name)
tom=Cat("Tom",20)
print(tom)