import pymysql
conn=pymysql.connect(host="127.0.0.1",port=3306,user="root",password="root",db="py")
cursor=conn.cursor()
#查询语句
select_sql="select * from User where username=%s and password=%s"
result=cursor.execute(select_sql,["张三","123456789"])
if result==1:
    print(type(cursor.fetchall()))
#插入语句：
insert_sql="insert into User(username,password) values(%s,%s)"
cursor.execute(insert_sql,["程起","123456789"])
conn.commit()
selectAll="select * from User"
# 更新语句
update_sql="update User set password=%s where username=%s"
cursor.execute(update_sql,["15465","张三"])
conn.commit()
cursor.execute(selectAll)
# print(cursor.fetchall())
print(cursor.fetchmany(2))