#模拟鼠标的操作：
from selenium.webdriver.common.action_chains import ActionChains
from selenium import  webdriver
driver=webdriver.Chrome("E:\chromedriver_win32\chromedriver.exe")
driver.get("https://www.baidu.com/")
element=driver.find_element_by_css_selector("a[name='tj_briicon']")
# print(element.get_attribute("outerHTML"))
act=ActionChains(driver)
#鼠标悬停
# act.move_to_element(element).perform()
#鼠标单击
act.context_click(element).perform()
#鼠标双击
# act.double_click(element).perform()

#alert点击
# driver.get("file:///C:/Users/ASUS/Desktop/temp/index2.html")
# elem_alert=driver.find_element_by_css_selector("#b1")
# elem_alert.click()
# driver.switch_to.alert.accept()
#确认键：
# elem_confirm=driver.find_element_by_css_selector("#b2")
# elem_confirm.click()
# # 点击确认键
# # driver.switch_to.alert.accept()
# # 点击取消键
# driver.switch_to.alert.dismiss()
#提交键
# elem_prompt=driver.find_element_by_css_selector("#b3")
# elem_prompt.click()
# #获得弹窗对象：
# alert=driver.switch_to.alert
# # 点击确认
# # alert.send_keys("我是大帅哥")
# # alert.accept()
# # 点击取消
# alert.dismiss()