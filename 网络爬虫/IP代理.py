import requests

proxy={"http":"http://122.51.210.221"}
# proxy={"http":"https://45.77.8.148"}
headers={"User-Agent": "Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Mobile Safari/537.36",
}
r=requests.get("https://www.baidu.com",proxies=proxy,headers=headers)
print(r.content.decode())