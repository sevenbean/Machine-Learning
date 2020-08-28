import json
books = [
    {
        "title":"坏蛋是怎样炼成的1","price":9.8
    },
    {
        "title":"坏蛋是怎样炼成的2","price":9.9
    }
 ]
#将内容dumps成字符串数组
json_str=json.dumps(books,ensure_ascii=False)
print(json_str)
print(type(books))
print(type(json_str))

with open("a.json", "w") as fp:
    json.dump(books,fp,ensure_ascii=False)