class Tools:
    score=0
    def __init__(self,name):
        self.name=name
        Tools.score+=1
    @staticmethod
    def game_help():
        print("游戏帮助")
    @classmethod
    def getScore(cls):
        print("获得的积分：%d"%(cls.score))
    def playgame(self):
        print("%s开始玩游戏"%(self.name))

chuizi=Tools("铁锤")
langtou=Tools("榔头")
Tools.getScore()
Tools.game_help()
chuizi.playgame()