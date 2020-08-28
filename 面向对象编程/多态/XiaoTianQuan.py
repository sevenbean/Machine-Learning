from 面向对象编程.多态.Dog import Dog


class XiaoTianQuan(Dog):
    def playGame(self):
        print("%s是一只神犬,在蹦蹦哒哒的玩耍....."%(self.name))