class Tools:
    count=0
    def __init__(self,name):
        self.name=name
        # 类属性
        Tools.count+=1
    @classmethod
    def show_count(cls):
        print("工具的数量%d"%(Tools.count))

tool1=Tools("铁锤")
tool2=Tools("榔头")
tool3=Tools("菜刀")
Tools.show_count()