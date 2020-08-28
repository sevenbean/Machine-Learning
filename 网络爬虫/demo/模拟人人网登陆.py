import requests
def getHtmlText(url,headers,data,url1):
    try:
        sess=requests.session()
        r=sess.post(url,headers=headers,data=data)
        r=sess.get(url1,headers=headers)
        return r.text
    except:
        pass
if __name__=="__main__":
    url="http://www.renren.com/PLogin.do"
    url1="http://www.renren.com/974760962"
    data={"email":"18868718660","password":"cwj330127"}
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"}
    print(getHtmlText(url,header,data,url1))