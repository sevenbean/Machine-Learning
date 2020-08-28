import requests
from bs4 import BeautifulSoup
import bs4

def getHtmlText(url,headers):
    try:
        r=requests.get(url,headers)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        return "连接出错"
def FillUnivList(text):
    University_Rank=[]
    soup=BeautifulSoup(text,"html.parser")
    for tr in soup.find_all("tbody","hidden_zhpm")[0].children:
        University=[]
        if isinstance(tr,bs4.element.Tag):
            tds=tr.find_all("td")
            University.append(tds[0].string)
            University.append(tds[1].string)
            University.append(tds[2].string)
        University_Rank.append(University)
    return University_Rank
def printUniversityRank(University_Rank,num):
    University_Rank=University_Rank[1:num+1]
    tplt="{0:^10}\t{1:{3}^10}\t{2:^10}"
    print(tplt.format("排名","学校名称","所在地",chr(12288)))
    print()
    for i in University_Rank:
        print(tplt.format(i[0],i[1],i[2],chr(12288)))

if __name__=="__main__":
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}
    requests_url="http://www.zuihaodaxue.com/zuihaodaxuepaiming2019.html"
    printUniversityRank(FillUnivList(getHtmlText(requests_url,headers)),100)
