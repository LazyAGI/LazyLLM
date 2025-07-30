import pandas as pd
import sqlite3

def convert_excel_to_sqlite(xlsx_path, sqlite_path, table_name="medical_visits"):
    df = pd.read_excel(xlsx_path)
    conn = sqlite3.connect(sqlite_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"[✓] 转换完成：{xlsx_path} -> {sqlite_path}，表名：{table_name}")

if __name__ == "__main__":
    convert_excel_to_sqlite(
        xlsx_path="/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs/blackfriday.xlsx",
        sqlite_path="/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs/blackfriday.db",
        table_name="blackfriday"
    )
