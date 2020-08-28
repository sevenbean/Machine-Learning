import  requests
import os
def getdata(url,headers):
    try:
        r=requests.get(url,headers=headers)
        r.raise_for_status()
        return r.content
    except:
        return "链接异常"
if __name__=="__main__":
    headers={"User-Agent":"Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Mobile Safari/537.36"}
    text=getdata("https://img12.360buyimg.com/pop/s590x470_jfs/t1/146751/28/2538/90808/5f07cd76E2e0ce035/b7ae0b95df571ff6.jpg",headers)
    if not os.path.exists("../one.jpg"):
        with open("../one.jpg", "wb") as file:
            file.write(text)
            file.close()
            print("写入完毕")
    else:
        print("文件已经存在")