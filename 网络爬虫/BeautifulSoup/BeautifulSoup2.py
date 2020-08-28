import re
from bs4 import BeautifulSoup
import requests
r=requests.get("http://python123.io/ws/demo.html")
soup=BeautifulSoup(r.text,"html.parser")
print(soup.prettify())
#读取这个文件中的所有标签：
for tag in soup.find_all(True):
    print(tag.name)
#查找所有的a标签
for i in soup.find_all("a"):
    print(i.attrs["href"])
# 查找a,b标签
for i in soup.find_all(["a","b"]):
    print(i)
#查找所有以B开头的字符串
for i in soup(re.compile("b")):
    print(i.name)
# 查找标签为P，属性值为"title"的数据
for i in soup.find_all("a","py1"):
    print(i)
#查找class="py1"的标签
for i in soup.find_all(id="link1"):
    print(i.name)
print(soup.find_next_sibling("p","title"))
print("*"*50)
#查找P标签的父标签
for i in soup.p.find_parents():
    print(i.name)
print(soup.p.find_next_sibling())
print("*"*20)
print(soup.p.string)
for i in soup.find_all("p","course")[0].children:
    print(i)