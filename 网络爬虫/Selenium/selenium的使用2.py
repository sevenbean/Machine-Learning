from selenium import webdriver
wd=webdriver.Chrome("E:\chromedriver_win32\chromedriver.exe")
wd.get("file:///C:/Users/ASUS/Desktop/temp/index.html")
#根据标签值选择：
element=wd.find_elements_by_css_selector("div")
for i in element:
    print(i.text)
#根据id值选择：
element_id=wd.find_element_by_css_selector("#searchtext")
element_id.send_keys("python")
#根据class值来选择：
element_class=wd.find_elements_by_css_selector(".raise")
for i in element_class:
    print(i)
print("*"*20)
#通过子元素选择：
element_son=wd.find_elements_by_css_selector("div>#inner11>span")
print(element_son)
#通过元素的属性来选择：
element_attrs=wd.find_elements_by_css_selector('[href="https://www.cnblogs.com/liuhui0308/"]')
print(element_attrs)
#通过组合的方式来选择：
element_zhuhe=wd.find_elements_by_css_selector(".raise,.wolf")
for i in element_zhuhe:
    print(i.text)
print("*-"*20)
element_paixu=wd.find_element_by_css_selector("body .raise:nth-child(2)")
print(element_paixu.get_attribute("innerHTML"))
print(element_paixu.get_attribute("outerHTML"))
print()
