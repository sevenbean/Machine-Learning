import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Mobile Safari/537.36"}

r=requests.get("https://www.ip138.com/iplookup.asp?ip=45.76.107.10&action=2",headers=headers)
r.encoding=r.apparent_encoding
print(r.text)
