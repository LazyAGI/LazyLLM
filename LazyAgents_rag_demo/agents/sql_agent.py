# text2sql_agent.py
import sys
sys.path.append("/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo")  # 添加项目根路径

import lazyllm
from tools.text2sql_tool import create_text2sql_pipeline

# 数据库路径
DB_PATH = "/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs/blackfriday.db"

# Prompt 模板
PROMPT = """你可以访问一个名为 blackfriday 的数据库表，包含以下字段（列）：

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
"""

if __name__ == "__main__":
    # 创建包含生成 + 执行的 pipeline
    pipeline = create_text2sql_pipeline(
        #prompt=PROMPT,
        source="qwen",
        db_path=DB_PATH
    )

    # 启动 WebModule 服务
    #lazyllm.WebModule(pipeline, port=range(23461, 23470)).start().wait()

