try:
    num=int(input("请输入一个整数："))
    div=8/num
    print(div)
except ZeroDivisionError:
    print("除数不能为0")
except ValueError:
    print("请输入一个整数")

