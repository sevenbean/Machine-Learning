import requests
from bs4 import BeautifulSoup
import  bs4
import os
def getHtmlText(url,headers):
    try:
        r=requests.get(url,headers=headers)
        r.raise_for_status()
        r.encoding="UTF-8"
        return r.text
    except:
        return "链接失效"
def getUrlList(text):
    soup=BeautifulSoup(text,"html.parser")
    a_list=[]
    for i in soup.find_all("div",attrs={"class":"book-mulu"})[0]:
        if isinstance(i,bs4.element.Tag):
           for item in i.find_all("a"):
               list_item = []
               list_item.append(item["href"])
               list_item.append(item.string)
               a_list.append(list_item)

    return a_list
def requestAllUrl(a_list,headers):
    url="http://www.shicimingju.com"

    try:
            print(url+a_list[0])
            text=getHtmlText(url+a_list[0],headers)
            soup=BeautifulSoup(text,"html.parser")
            wenben=soup.find_all("div",attrs={"class":"chapter_content"})[0].text
            if not os.path.exists("../Books"):
                os.makedirs("Books")
            else:
                print(a_list[1])
                with open("Books/"+a_list[1]+".txt","w",encoding="UTF-8") as fl:
                        fl.write(a_list[1]+"\n"+wenben)
    except:
                pass

if __name__=="__main__":
    url="http://www.shicimingju.com/book/sanguoyanyi.html"
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}
    a_list=getUrlList(getHtmlText(url,header))
    for i in range(8):
        requestAllUrl(a_list[i], header)

