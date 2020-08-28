try:
    num=int(input("请输入一个整数"))
    div=10/num
    print(div)
except Exception as e:
    print("异常：%s"%(e))
else:
    #只有在没有异常的情况下才会被执行
    print("没有异常")
finally:
    print("无论有没有异常都会执行成功")
