import sqlite3

# 指定你的 SQLite 数据库路径
db_path = "/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs/blackfriday.db"

# 连接数据库
conn = sqlite3.connect(db_path)

# 创建 cursor
cursor = conn.cursor()

# 执行查询
cursor.execute("SELECT Gender FROM blackfriday WHERE User_ID = 1000001")

# 获取结果
results = cursor.fetchall()
print("查询结果：", results)

# 关闭连接
cursor.close()
conn.close()
