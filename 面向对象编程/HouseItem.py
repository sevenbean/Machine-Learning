class HouseItem:
    def __init__(self,name,area):
        self.name=name
        self.area=area
    def __str__(self):
        return "%s占地%.2f平米"%(self.name,self.area)
