import requests
import json
def getData(url,headers,data):
    try:
        # response=requests.get(url,headers)
        response=requests.get(url,headers=headers,params=data)
        response.raise_for_status()
        response.encoding=response.apparent_encoding
        return response.text
    except:
        return "产生异常"

if __name__=="__main__":
    headers={"User-Agent":"Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Mobile Safari/537.36"}
    data=getData("http://www.baidu.com/s",headers,{"wd":"python","ie":"UTF-8"})
    print(data)