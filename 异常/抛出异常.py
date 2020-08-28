def input_psw():
    psw=input("请输入密码：")
    if len(psw)>=8:
        return  psw
    #自定义异常类，然后将其抛出
    ex=Exception("密码长度不够")
    raise  ex

try:
    print(input_psw())
except Exception as exception:
    print(exception)