import sqlite3
from lazyllm import ChatPrompter, OnlineChatModule
from lazyllm.tools import SqlCall, SqlManager
import json

# 📌 数据库路径
db_path = "/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs/blackfriday.db"

# 提取 SQLite 表结构
def extract_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(blackfriday);")
    schema_lines = []
    for row in cursor.fetchall():
        schema_lines.append(f"{row[1]} ({row[2]})")
    conn.close()
    return "blackfriday 表包含列:\n- " + "\n- ".join(schema_lines)

# 构建 text2sql pipeline
def create_text2sql_pipeline(source: str, db_path: str):
    schema = extract_schema(db_path)

    # Prompt 提示词
    prompt = f"""
你可以访问一个名为 blackfriday 的数据库表，包含以下字段（列）：

- User_ID：用户ID（整数）
- Product_ID：商品ID（字符串）
- Gender：性别（'M' 或 'F'）
- Age：年龄区间，例如 '26-35'、'55+'
- Occupation：职业编号（整数）
- City_Category：所在城市分类（'A', 'B', 'C'）
- Stay_In_Current_City_Years：在当前城市居住年数，例如 '2', '4+', '1'
- Marital_Status：婚姻状态，0 表示未婚，1 表示已婚
- Product_Category_1：商品一级类别（整数）
- Product_Category_2：商品二级类别（整数，可空）
- Product_Category_3：商品三级类别（整数，可空）
- Purchase：用户购买金额（整数）

请根据用户自然语言问题，生成相应的 SQL 查询语句。
{schema}
"""

    # 初始化 prompter 和 LLM
    prompter = ChatPrompter(prompt)
    sql_llm = OnlineChatModule(source=source).prompt(prompter)

    tables_info_dict = {
        "tables": [
            {
                "name": "blackfriday",
                "columns": [
                    {"name": "User_ID", "data_type": "integer", "is_primary_key": True},
                    {"name": "Product_ID", "data_type": "string"},
                    {"name": "Gender", "data_type": "string"},
                    {"name": "Age", "data_type": "string"},
                    {"name": "Occupation", "data_type": "integer"},
                    {"name": "City_Category", "data_type": "string"},
                    {"name": "Stay_In_Current_City_Years", "data_type": "string"},
                    {"name": "Marital_Status", "data_type": "integer"},
                    {"name": "Product_Category_1", "data_type": "integer"},
                    {"name": "Product_Category_2", "data_type": "integer"},
                    {"name": "Product_Category_3", "data_type": "integer"},
                    {"name": "Purchase", "data_type": "integer"}
                ]
            }
        ]
    }

    sql_tool = SqlManager(
        db_type="sqlite",
        user="",
        password="",
        host="",
        port=0,
        db_name=db_path,
        tables_info_dict=tables_info_dict
    )

    sql_agent = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=False)

    # ✅ pipeline 主函数
    def pipeline(user_input: str):
        execution = sql_agent(user_input)
        print("🔥 原始返回:", execution)

        # ✅ 如果返回是 JSON 字符串，先尝试解析成对象
        if isinstance(execution, str):
            try:
                execution = json.loads(execution)
                print("✅ 已成功解析 JSON 字符串为 Python 对象")
            except json.JSONDecodeError as e:
                return f"❌ 返回结果无法解析为 JSON：{str(e)}\n\n原始内容：\n```\n{execution}\n```"

        try:
            if isinstance(execution, list) and all(isinstance(row, dict) for row in execution):
                if execution:
                    columns = list(execution[0].keys())
                    output = "📊 查询结果：\n"
                    output += " | ".join(columns) + "\n"
                    output += "-" * len(output) + "\n"
                    for row in execution:
                        output += " | ".join(str(row[col]) for col in columns) + "\n"
                    markdown_table = output
                else:
                    markdown_table = "✅ SQL 执行成功，但结果为空。"

            elif isinstance(execution, dict):
                if execution.get("error") is None and execution.get("columns"):
                    columns = execution["columns"]
                    rows = execution["rows"]
                    output = "📊 查询结果：\n"
                    output += " | ".join(columns) + "\n"
                    output += "-" * len(output) + "\n"
                    for row in rows:
                        output += " | ".join(str(item) for item in row) + "\n"
                    markdown_table = output
                else:
                    markdown_table = f"❌ 查询失败: {execution.get('error') or '无列信息返回'}"

            else:
                markdown_table = f"⚠️ 查询返回非预期格式：\n```json\n{json.dumps(execution, ensure_ascii=False, indent=2)}\n```"

        except Exception as e:
            markdown_table = f"❌ 查询异常: {str(e)}"

        return (
            "### 📌 LLM 回复（自动生成的 SQL 已执行）\n\n"
            "👇 查询结果（Markdown 表格）：\n\n"
            f"{markdown_table}"
        )


    return pipeline

# ✅ 可选测试：验证数据库连接
if __name__ == "__main__":
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM blackfriday")
    print("🔥 blackfriday 表行数:", cursor.fetchone())
    conn.close()
