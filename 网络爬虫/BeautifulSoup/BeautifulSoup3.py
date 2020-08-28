import requests
from bs4 import BeautifulSoup
html = """
<html>
<head><title>标题</title></head>
<body>
 <p class="title" name="dromouse"><b>标题</b></p>
 <div name="divlink">
  <p>
   <a href="http://example.com/1" class="sister" id="link1">链接1</a>
   <a href="http://example.com/2" class="sister" id="link2">链接2</a>
   <a href="http://example.com/3" class="sister" id="link3">链接3</a>
  </p>
 </div>
 <p></p>
 <div name='dv2'></div>
</body>
</html>
"""
soup=BeautifulSoup(html,"html.parser")
#通过标签选择器，选择
print(soup.select("title")[0])
# 通过标签逐层查找
print(soup.select("html head title"))
#通过id选择器选择
print(soup.select("#link1"))
#通过class标签来寻找
print(soup.select(".sister"))
#通过组合进行选择
print(soup.select("p #link1"))
print("*"*20)
print(soup.select("p .sister"))
print("*"*20)
print(soup.select_one("head >title"))
print(soup.select("p >#link2"))
print(soup.select("p >a:nth-of-type(2)"))
print("*"*50)
print(soup.select("#link1~#link3"))
print("*"*50)
print(soup.select("#link1~.sister"))
print("*"*20)
print(soup.select("[name]"))