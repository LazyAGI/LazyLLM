# tools/query_executor.py

import sqlite3
import pandas as pd

def run_query(sql: str, db_path="db/sales.db") -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])
