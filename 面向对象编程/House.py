class House:
    houseItem_list = []

    def __init__(self, houseType, area):
        self.houseType = houseType
        self.area = area

    def addHouseItem(self, houseItem):
        self.houseItem_list.append(houseItem)

    def __str__(self):
        for i in self.houseItem_list:
            self.area -= i.area
        return "户型:", self.houseType, "总面积：", self.area, "剩余面积：", self.area, "家具列表：%s" % (self.houseItem_list)
