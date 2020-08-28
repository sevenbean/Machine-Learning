from selenium import webdriver
wd=webdriver.Chrome("E:\chromedriver_win32\chromedriver.exe")
# #设置等待的时间
wd.implicitly_wait(60)
wd.get("https://www.baidu.com")
element=wd.find_element_by_id("kw")
element.send_keys("python")
#获取对应标签的属性值
print(element.get_attribute("class"))



