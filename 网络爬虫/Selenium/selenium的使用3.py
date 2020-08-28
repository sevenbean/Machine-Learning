from selenium import webdriver
wd=webdriver.Chrome("E:\chromedriver_win32\chromedriver.exe")
wd.get("file:///C:/Users/ASUS/Desktop/temp/index1.html")
elements=wd.find_elements_by_css_selector("#s_radio input")
elements[1].click()
for i in elements:
    print(i.get_attribute("outerHTML")+i.get_attribute("value"))
#多选框：
elements_checkboxs=wd.find_elements_by_css_selector("#s_checkbox input[checked='checked']")
for i in elements_checkboxs:
    print(i.get_attribute("outerHTML"))
    i.click()

wd.find_element_by_css_selector("#s_checkbox input[value='checkbox2']").click()