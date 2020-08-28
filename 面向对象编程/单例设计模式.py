#实现单例模式:不管创建多少次对象，都是同一个对象
class MusicPlayer(object):
    instance=None
    MusicPlayer_init_Flag=False
    # 调用new这个内置函数创建一个对象的应用，将创建的对象引用，放到init中去
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance=super().__new__(cls)
        return cls.instance
    # 进行单例化时：只进行一次初始化操作
    def __init__(self):
        #首先判断是否进行了实力化：
        if MusicPlayer.MusicPlayer_init_Flag:
            return
        MusicPlayer.MusicPlayer_init_Flag=True
        print("创建了一个对象")


musicplay1=MusicPlayer()
print(musicplay1)
musicplay2=MusicPlayer()
print(musicplay2)