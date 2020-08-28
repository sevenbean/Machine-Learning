import requests
class TiebaSpider:
    def __init__(self,TiebarName):
        self.urlList=[]
        self.TiebarName=TiebarName
        self.url="https://tieba.baidu.com/f?kw="+TiebarName+"&pn={}"
        pass
    def addUrl(self):
        for i in range(10):
            self.urlList.append(self.url.format(i*50))
    def save_html(self,pagenum,text):
        file=open(self.TiebarName+"第{}页.html".format(pagenum),"w",encoding="utf-8")
        file.write(text)
        print(file.name)
    def run(self):
        header={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}
        for url in self.urlList:
            response=requests.get(url,headers=header)
            self.save_html(self.urlList.index(url),response.text)
if __name__=="__main__":
    tieba=TiebaSpider("李毅")
    tieba.addUrl()
    tieba.run()