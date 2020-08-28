from selenium import  webdriver
from time import sleep
driver=webdriver.Chrome("E:\chromedriver_win32\chromedriver.exe")
driver.get("http://www.renren.com/SysHome.do")
username=driver.find_element_by_css_selector("input[id='email']")
password=driver.find_element_by_css_selector("input[id='password']")
login=driver.find_element_by_css_selector("input[id='login']")
username.send_keys("18868718660")
password.send_keys("cwj330127")
login.click()
driver.switch_to.window(driver.window_handles[0])
sleep(5)
element=driver.find_element_by_css_selector("a[class='app-link'] .app-title")
print(element.text)
