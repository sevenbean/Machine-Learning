from bs4 import BeautifulSoup
import requests
text=requests.get("http://python123.io/ws/demo.html").text
soup=BeautifulSoup(text,"html.parser")
print(soup.prettify())
#获取body的孩子标签
for i  in soup.body.children:
    if i is None:
        print(i)
    else:
        print(i.name)
print("-"*50)
for i in soup.body.contents:
    if i is None:
        print(i)
    else:
        print(i.name)
#获取兄弟节点
print("-"*50)
for i in soup.p.next_siblings:
    if i is None:
        print(i)
    else:
        print(i.name)
print("*"*50)
print(soup.a.parent.name)
print("*"*50)
for i in soup.a.parents:
    print(i.name)
