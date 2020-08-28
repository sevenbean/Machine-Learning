from 面向对象编程.多态.Dog import Dog
from 面向对象编程.多态.XiaoTianQuan import XiaoTianQuan


class People:
    def __init__(self,name):
        self.name=name
    def game_with_dog(self,dog):
        print("%s和%s在一起玩耍"%(self.name,dog.name))
        dog.playGame()

# wangcai=Dog("旺财")
wangcai=XiaoTianQuan("飞天旺财")
erlangsheng=People("二郎神")
erlangsheng.game_with_dog(wangcai)

