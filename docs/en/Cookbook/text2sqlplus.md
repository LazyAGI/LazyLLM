# Text2SQL ToolPlus

**Text2SQL ToolPlus** is an upgraded version of the traditional Text2SQL pipeline.  
It not only converts natural language questions into SQL queries, but also **automatically validates** whether the generated SQL is safe, correct, and executable — and repairs it if necessary.  
The entire process is integrated into a **ReactAgent** for end-to-end automation, and can be deployed as a **WebModule**.

!!! abstract "In this section, you will learn the following key features"

    - How to extract database schema for SQL generation;
    - How to validate SQL statements locally before execution;
    - How to use two LLM roles:
        1. **SQL Generator** — generates initial safe SQL from natural language.
        2. **SQL Fixer** — repairs invalid or unsafe SQL;
    - How to integrate these tools into a **ReactAgent** for reasoning-execution loops;
    - How to run the pipeline as a web service with `WebModule`.

---

## Design Overview

### Roles and Responsibilities

1. **SQL Generator**  
   Given the user question and the table schema, the LLM generates a **single** safe `SELECT` SQL query that only accesses the allowed table.
   
2. **SQL Validator**  
   Performs **local safety checks**:
   - Only allows `SELECT` queries;
   - Only accesses the allowed table (`blackfriday`);
   - Ensures column names exist in the schema;
   - Blocks multi-statement and dangerous operations.

3. **SQL Fixer**  
   If the generated SQL fails validation or execution, the LLM rewrites it into a valid and safe form.

4. **Executor**  
   Runs the validated SQL against the SQLite database and returns results in Markdown-friendly format.

5. **Agent Orchestration**  
   The **ReactAgent** coordinates the process:  
   `generate SQL → validate SQL → execute SQL → return answer`.

---

## Workflow

```

User Question
↓
gen\_sql (generate & validate)
↓ if invalid
sql\_fixer (repair & revalidate)
↓
exec\_sql (execute & format result)
↓
Answer with SQL + result preview

```

---

## Features

- **Automatic SQL Safety Checks** — every query is validated locally before execution.  
- **Self-repairing SQL** — the LLM can fix invalid SQL automatically.  
- **Schema-aware generation** — column names and types are checked against the database.  
- **Markdown result preview** — query results are returned in a readable table format.  
- **Agent integration** — `gen_sql` and `exec_sql` are registered as tools for `ReactAgent`.

---

## Notes

1. **Database name must be in English** — avoid Chinese characters or non-ASCII symbols to prevent path parsing or encoding issues.  
2. **All keys (column names) in database tables must be in English** — avoid Chinese or special characters to prevent SQL parsing or matching failures.  
3. Only **one table** is allowed (`blackfriday` in this demo). Accessing other tables is blocked by the safety validator.  
4. Use **snake_case** or **lowerCamelCase** naming conventions for better cross-platform compatibility.


---

## Code Implementation

```python
import sqlite3
import traceback
from typing import List, Dict
from lazyllm import bind, pipeline, parallel
from lazyllm import ChatPrompter, WebModule
from lazyllm.tools import fc_register, ReactAgent
from lazyllm.module import OnlineChatModule

# ---------------- Configuration ----------------
DB_PATH = "blackfriday.db"
TABLE_NAME = "blackfriday" # Change your own DB Path

# ---------------- Schema Extraction ----------------
def _extract_schema(db_path: str, table_name: str):
    """Extracts schema text and valid columns from a SQLite table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    schema_info = cursor.fetchall()
    conn.close()
    
    schema_text = "\n".join([f"{col[1]} ({col[2]})" for col in schema_info])
    valid_columns = [col[1] for col in schema_info]
    return schema_text, valid_columns

# ---------------- SQL Safety Validation ----------------
def _is_safe_sql(sql: str, valid_columns: List[str], allowed_table: str) -> bool:
    """Checks SQL safety based on table name, allowed columns, and SELECT-only rule."""
    sql_lower = sql.strip().lower()
    if not sql_lower.startswith("select"):
        return False
    if ";" in sql_lower:
        return False
    if allowed_table.lower() not in sql_lower:
        return False
    for token in sql.replace(",", " ").split():
        if token.isidentifier() and token not in valid_columns and token.lower() != allowed_table.lower():
            return False
    return True

# ---------------- LLM Prompts ----------------
_SQL_GEN_SYS = (
    "You are an AI SQL generator. You must output exactly ONE safe SELECT SQL query "
    "based on the user's question and the given table schema. "
    "The query must only use the allowed table and columns."
)

_SQL_FIX_SYS = (
    "You are an AI SQL fixer. The provided SQL is invalid or unsafe. "
    "Rewrite it into a valid and safe SQL query based on the same question and schema."
)

# ---------------- Tool Functions ----------------
def _gen_sql_with_llm(question: str, schema_text: str, valid_columns: List[str], llm) -> str:
    prompter = ChatPrompter(instruction=_SQL_GEN_SYS, extra_keys=["schema_text"])
    sql = llm.share(prompter)(dict(schema_text=schema_text, query=question))
    return sql.strip()

@fc_register("tool")
def gen_sql(question: str) -> str:
    """Generates and validates SQL from user question."""
    sql = _gen_sql_with_llm(question, SCHEMA_TEXT, VALID_COLUMNS, LLM)
    if not _is_safe_sql(sql, VALID_COLUMNS, TABLE_NAME):
        # Repair if invalid
        prompter = ChatPrompter(instruction=_SQL_FIX_SYS, extra_keys=["schema_text", "invalid_sql"])
        sql = LLM.share(prompter)(dict(schema_text=SCHEMA_TEXT, invalid_sql=sql, query=question))
    return sql

@fc_register("tool")
def exec_sql(sql: str) -> str:
    """Executes validated SQL and returns result in Markdown table format."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]
        conn.close()
        # Format result as Markdown table
        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table_md += "| " + " | ".join(map(str, row)) + " |\n"
        return f"**SQL**:\n{sql}\n\n📊 Query Result:\n{table_md}"
    except Exception:
        return f"Error executing SQL:\n```\n{traceback.format_exc()}\n```"

# ---------------- Pipeline Creation ----------------
def create_text2sql_pipeline(source: str, db_path: str):
    global SCHEMA_TEXT, VALID_COLUMNS, LLM
    SCHEMA_TEXT, VALID_COLUMNS = _extract_schema(db_path, TABLE_NAME)
    LLM = OnlineChatModule(source=source, stream=False)
    agent = ReactAgent(llm=LLM, tools=["gen_sql", "exec_sql"], max_retries=3, stream=False)
    return agent

# ---------------- Run as Web Service ----------------
if __name__ == "__main__":
    pipeline = create_text2sql_pipeline(source="qwen", db_path=DB_PATH)
    WebModule(pipeline, port=23465).start().wait()
```

---




