from 面向对象编程.House import House
from 面向对象编程.HouseItem import HouseItem

ximengsi=HouseItem("席梦思",4.0)
chugui=HouseItem("橱柜",5.0)
bieshu=House("别墅",80.0)
bieshu.addHouseItem(ximengsi)
bieshu.addHouseItem(chugui)
print(bieshu.__str__())