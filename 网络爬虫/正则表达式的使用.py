import re
pat=re.compile(r'[1-9]\d{5}')
#返回一个匹配的结果
# match=pat.search("BIT100081")
match=pat.search("Zip145487")
print("返回匹配的结果："+match.group(0))
print("匹配的字符串："+match.string)
print("-"*20)
# 是否匹配字符串：从一个字符串的起始位置起匹配正则表达式，返回match对象
match1=pat.match("145487 BIT")
if match1 is None:
    print("空类型")
else:
    print("返回匹配的结果："+match1.group(0))
# 返回匹配的字符串列表
match2=pat.findall("VIP845678")
print(match2)
print("-"*50)
#将不匹配的字符串返回成一个列表
match3=pat.split("BIT488555 tIN458745",maxsplit=1)
print(match3)
print("-"*50)
#将匹配的字符串替换成前面的字符串
match4=pat.sub("woaini","jdah154868")
print(match4)