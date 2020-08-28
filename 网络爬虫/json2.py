import json
json_str = [{"title": "坏蛋是怎样炼成的1", "price": 9.8}, {"title": "坏蛋是怎样炼成的2", "price": 9.9}]
books=json.loads(json.dumps(json_str,ensure_ascii=False))
print(type(json_str))
for i in books:
    print("书名："+i["title"]+"价格",i["price"])
#直接从文件中读取
with open("a.json","r") as fp:
    books=json.load(fp)
    for book in books:
        print("书名：" + i["title"] + "价格", i["price"])