# Text2SQL·ToolPlus —— 自动校验与修复的安全可执行 Text2SQL

本文实现一个升级版 **Text2SQL** 工具：在将自然语言转换为 SQL 之前与之后，均进行**本地安全校验**；若不通过，则自动进入 **SQL 修复回路**，直至产出**单条、仅 `SELECT`、可执行**且**安全**的 SQL，并执行查询，将结果以 Markdown 的简洁表格形式返回。整体能力基于 **LazyLLM** 的工具化与 ReAct Agent 流水线实现。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

```
- 如何使用 [OnlineChatModule][lazyllm.module.OnlineChatModule] 构建 SQL 生成器与修复器；
- 如何用 [fc_register][lazyllm.tools.fc_register] 将能力注册为可被 Agent 调用的工具；
- 如何设计本地 SQL 安全审查：只允许 SELECT、限制表名、验证列名、拦截多语句与危险关键字；
- 如何在工具内实现“生成 → 校验 → 修复”的**自洽闭环**；
- 如何封装成 `create_text2sql_pipeline`，对外暴露统一的 Text2SQL 可调用接口；
- 如何使用 [ReactAgent][lazyllm.tools.ReactAgent] 将“生成 SQL / 执行 SQL”组合为端到端工作流；
- 如何用 [WebModule][lazyllm.WebModule] 启动为本地 Web 服务。
```

---

## 设计思路

### 角色与职责

1. **SQL 生成器（Generator）**
   基于用户问题与表结构，产出**单条** `SELECT` SQL（禁止多语句与任何写操作/危险操作）。
2. **SQL 修复器（Fixer）**
   当本地安全审查或执行失败时，利用“问题 + 表结构 + 上一条 SQL + 错误信息”进行**定向修复**，再次生成候选 SQL。
3. **本地安全审查（Local Guard）**
   不依赖模型，在落地前进行严谨的**语法与安全检查**：仅 `SELECT`、仅 `blackfriday` 表、列名在白名单内（大小写不敏感），禁止多语句与危险关键字。
4. **执行器（Executor）**
   基于已校验通过的 SQL 执行查询，返回列名与数据行，并将查询结果转成易读的 Markdown 文本。

### 工作流（ReAct + 工具路由）

```
用户问题
   ↓
gen_sql（生成或修复 SQL） → 本地安全校验失败？→ 是 → 进入修复回路（最多重试 N 次）
   ↓ 否
exec_sql（执行） → 结果转 Markdown → 返回
```

---

## 安全策略与健壮性

* **只允许 `SELECT`**：任何 `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA` 等均直接拦截。
* **仅访问指定表**：强制限制在 `blackfriday` 表；`FROM/JOIN` 的表名大小写不敏感且去引号比对。
* **列名白名单**：从 `PRAGMA table_info` 动态加载列集合；疑似未知列会触发修复回路。
* **拒绝多语句**：仅允许单条 SQL；末尾分号允许但不允许中间出现第二个分号。
* **可读性输出**：将执行结果转为简洁 Markdown，便于前端展示或直接粘贴到文档。

---

## 注意事项

1. **数据库文件名必须为英文**，不可使用中文字符（避免路径解析或编码问题）。
2. **数据库表中的所有字段名（Key）必须为英文**，不可使用中文或特殊字符（防止 SQL 解析与匹配失败）。
3. 建议遵循 **snake\_case** 或 **lowerCamelCase** 命名规范，确保跨平台与多数据库引擎的兼容性。

---

## 代码实现

### 1）数据库与表结构

* 通过 `_extract_schema` 读取 `blackfriday` 表结构，得到**文本描述**与**列名清单**；
* 在 `create_text2sql_pipeline` 中完成初始化与缓存，便于多次调用复用。

### 2）提示词设计

* **\_SQL\_GEN\_SYS**：指导生成**单条** `SELECT` SQL，仅访问 `blackfriday` 表，禁止任何解释与 Markdown 包裹；
* **\_SQL\_FIX\_SYS**：根据上一条 SQL 与错误原因进行**定向修复**，同样只返回 SQL 本体。

### 3）安全校验器

* `_is_safe_sql` 实现本地审查：校验操作类型、表名是否符合、列名是否在白名单、是否包含危险关键字与多语句。

### 4）工具注册

* `@fc_register("tool") gen_sql`：**先生成后校验，不通过则修复**，最多重试 `_MAX_RETRY` 次；
* `@fc_register("tool") exec_sql`：执行通过校验的 SQL，返回结构化结果。

### 5）对外统一入口

* `create_text2sql_pipeline(source, db_path)` 返回一个可调用的 `pipeline(user_input: str)`：

  * 内部用 `ReactAgent` 作为规划器，串联工具；
  * 结果阶段再次**显式执行**并包装 Markdown，输出“模型回复 + SQL + 结果”的直观视图。

### 6）启动为 Web 服务（可选）

* 直接在 `__main__` 中创建 pipeline 并以 `WebModule` 启动，便于 HTTP 方式接入。

---

## 启动应用

```python
pipeline = create_text2sql_pipeline(
    source="qwen",                # 规划 LLM 源
    db_path=DB_PATH               # SQLite 数据库路径
)
WebModule(pipeline, port=range(23461, 23470)).start().wait()
```

---


## 完整代码

<details>
<summary>点击查看完整代码</summary>

````python
# text2sql_toolplus.py 
# -*- coding: utf-8 -*-
import re
import sqlite3
from typing import List, Tuple, Dict, Any

from lazyllm.module import OnlineChatModule
from lazyllm.tools import fc_register
from lazyllm import WebModule, ChatPrompter
from lazyllm.tools import ReactAgent
import lazyllm

# 数据库路径
DB_PATH = "YOUR-OWN-PATH"

# ============== 基础工具 ==============

def _extract_schema(db_path: str) -> Tuple[str, List[str]]:
    """
    读取 blackfriday 表结构，返回文本描述与列名列表。
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(blackfriday);")
    cols, lines = [], []
    for row in cur.fetchall():
        cols.append(row[1])
        lines.append(f"{row[1]} ({row[2]})")
    conn.close()
    schema_text = "blackfriday 表包含列:\n- " + "\n- ".join(lines)
    return schema_text, cols


def _is_safe_sql(sql: str, allowed_table: str, allowed_cols: List[str]) -> Tuple[bool, str]:
    """
    做大小写不敏感的 SQL 安全校验（仅允许 SELECT，且只访问指定表；列名大小写不敏感）。
    """
    raw = sql.strip().strip(";")
    s = raw.lower()

    # 仅允许 SELECT
    if not s.startswith("select"):
        return False, "只允许 SELECT 查询。"

    # 禁多语句（允许末尾一个 ;）
    if ";" in raw[:-1]:
        return False, "不允许多条 SQL 语句。"

    # 禁危险关键字
    forbidden = [" insert ", " update ", " delete ", " drop ", " alter ",
                 " create ", " attach ", " detach ", " pragma "]
    if any(tok in f" {s} " for tok in forbidden):
        return False, "检测到潜在写操作或危险语句。"

    # 表名检查：大小写不敏感 + 去引号
    tbls = re.findall(r"\bfrom\s+([`\"']?[A-Za-z0-9_]+[`\"']?)", raw, flags=re.IGNORECASE) \
         + re.findall(r"\bjoin\s+([`\"']?[A-Za-z0-9_]+[`\"']?)", raw, flags=re.IGNORECASE)
    tbls = [t.strip().strip("`\"'") for t in tbls]
    if not tbls:
        return False, "未检测到 FROM/JOIN 的表名。"
    if any(t.lower() != allowed_table.lower() for t in tbls):
        return False, f"只允许访问表 {allowed_table}。"

    # 列名检查：大小写不敏感；放行 SQL 关键字/常见函数/数字/表名本身
    col_tokens = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", s)
    keywords = {
        "select","from","where","group","by","order","limit","asc","desc","and","or","not",
        "on","inner","left","right","join","having","distinct","as","in","is","null","between","like"
    }
    funcs = {"avg","sum","min","max","count","upper","lower","substr","strftime","cast","coalesce","round"}
    allowed_lower = {c.lower() for c in allowed_cols}
    suspicious = []
    for tok in set(col_tokens):
        if tok in keywords or tok in funcs or tok == allowed_table.lower() or re.match(r"^\d+$", tok):
            continue
        # 可能是别名，这里保守处理：不在列集合就标记为可疑，交给修复器兜底
        if tok not in allowed_lower:
            suspicious.append(tok)
    if suspicious:
        return False, f"疑似使用了不存在的列：{', '.join(sorted(suspicious))}"
    return True, ""


def _format_markdown(columns: List[str], rows: List[tuple]) -> str:
    """
    将查询结果转成简单 Markdown 文本（非严格表格，仅便于预览）。
    """
    if not columns:
        return "✅ SQL 执行成功（无结果集返回）。"
    header = " | ".join(columns)
    sep = "-" * max(3, len(header))
    if not rows:
        return f"📊 查询结果：\n{header}\n{sep}\n（结果为空）"
    lines = [f" | ".join(str(x) for x in r) for r in rows]
    return "📊 查询结果：\n" + header + "\n" + sep + "\n" + "\n".join(lines)

# ============== LLM 提示词 ==============

_SQL_GEN_SYS = """你是一个资深的 SQLite Text-to-SQL 助手。
目标：根据用户问题与提供的表结构，生成**单条**安全、正确、可执行的 SQL（只允许 SELECT）。
要求：
1）只访问 blackfriday 表，禁止其他表/ATTACH/PRAGMA/多语句；
2）只输出 SQL 代码本体，不要解释、不用 Markdown 代码块；
3）列名需精确，必要时使用聚合/分组/排序。
"""

_SQL_FIX_SYS = """你是一个 SQLite SQL 修复器。给你：
- 用户问题
- 表结构
- 上一条 SQL
- 错误或安全审核结论
请返回**一条修正后的可执行 SELECT SQL**，只访问 blackfriday 表，只输出 SQL 本体。
"""

_MAX_RETRY = 3

# ============== 全局运行态（由 create_text2sql_pipeline 初始化） ==============

_DB_PATH: str | None = None
_ALLOWED_COLS: List[str] = []
_ALLOWED_COLS_LOWER = set()
_SCHEMA_BLOCK = ""


def _new_llm(source: str):
    """
    简单封装，按需创建 OnlineChatModule。
    """
    return OnlineChatModule(source=source)

# ============== 注册为工具（供 ReactAgent 调用） ==============

@fc_register("tool")
def gen_sql(question: str, source: str = "qwen") -> str:
    """
    Generate or fix a single safe SELECT SQL for the blackfriday table.

    Args:
        question (str): User's natural-language question to convert into SQL.
        source (str): LLM provider used for generation and fixing. Defaults to "qwen".

    Returns:
        str: A single safe SELECT SQL statement (no Markdown, no explanations).
    """
    # 生成器与修复器：用系统提示词初始化
    llm_gen = _new_llm(source).prompt(ChatPrompter(_SQL_GEN_SYS))
    llm_fix = _new_llm(source).prompt(ChatPrompter(_SQL_FIX_SYS))

    last_sql, last_err = "", ""
    for attempt in range(_MAX_RETRY + 1):
        if attempt == 0:
            prompt = f"用户问题：{question}\n\n表结构：\n{_SCHEMA_BLOCK}\n\n只输出 SQL："
            candidate = llm_gen(prompt).strip()
        else:
            prompt = (
                f"用户问题：{question}\n\n表结构：\n{_SCHEMA_BLOCK}\n"
                f"上一条 SQL：{last_sql}\n错误/审核信息：{last_err}\n"
                f"请输出修正后的 SQL："
            )
            candidate = llm_fix(prompt).strip()

        safe, reason = _is_safe_sql(candidate, "blackfriday", _ALLOWED_COLS)
        if safe:
            return candidate
        last_sql, last_err = candidate, f"本地安全审查未通过：{reason}"

    raise ValueError(f"无法生成安全 SQL。最后错误：{last_err}")


@fc_register("tool")
def exec_sql(sql: str) -> Dict[str, Any]:
    """
    Execute a validated SELECT SQL against the blackfriday SQLite database.

    Args:
        sql (str): A single safe SELECT SQL statement to execute.

    Returns:
        Dict[str, Any]: Result with keys:
            - "columns" (List[str]): column names.
            - "rows" (List[List[Any]]): result rows.
            - "row_count" (int): number of rows.
    """
    if _DB_PATH is None:
        raise RuntimeError("DB 尚未初始化。请先调用 create_text2sql_pipeline。")

    safe, reason = _is_safe_sql(sql, "blackfriday", _ALLOWED_COLS)
    if not safe:
        raise ValueError(f"SQL 未通过安全审查：{reason}")

    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return {"columns": cols, "rows": rows, "row_count": len(rows)}
    finally:
        conn.close()

# ============== 对外主入口：保持原函数名与签名不变 ==============

def create_text2sql_pipeline(source: str, db_path: str):
    """
    返回一个可调用的 pipeline(user_input: str)。
    内部使用 ReactAgent + 工具(gen_sql/exec_sql)完成 Text2SQL。
    """
    global _DB_PATH, _ALLOWED_COLS, _ALLOWED_COLS_LOWER, _SCHEMA_BLOCK
    _DB_PATH = db_path
    schema_text, _ALLOWED_COLS = _extract_schema(db_path)
    _ALLOWED_COLS_LOWER = {c.lower() for c in _ALLOWED_COLS}  # 备份小写集合（目前 _is_safe_sql 内部自建，但保留以便将来扩展）
    _SCHEMA_BLOCK = f"{schema_text}\n（仅此一张表，禁止访问其他表）"

    # 规划 LLM（ReAct 计划/思考）；工具各自使用自己的 OnlineChatModule
    planner_llm = OnlineChatModule(source=source, stream=False)
    tools = ["gen_sql", "exec_sql"]
    agent = ReactAgent(planner_llm, tools, max_retries=6, return_trace=False, stream=False)

    def pipeline(user_input: str) -> str:
        """
        输入自然语言问题 → Agent 自动：
          1) gen_sql 生成或修复 SQL（含本地安全校验）
          2) exec_sql 执行
          3) 返回 Answer: ...（我们再做一次易读性包装）
        """
        res = agent(user_input)
        text = str(res).strip()

        # 尝试在最终回答里附上 SQL 与结果，便于前端显示
        try:
            sql = gen_sql(user_input, source=source)
            exec_res = exec_sql(sql)
            md = _format_markdown(exec_res.get("columns", []), exec_res.get("rows", []))
            return (
                "### 📌 LLM 回复（已生成并执行 SQL）\n"
                f"**SQL**：\n```\n{sql}\n```\n\n"
                f"{md}"
            )
        except Exception:
            # 补跑失败就退回 Agent 的自然语言回答
            if text.startswith("Answer:"):
                return text
            return "Answer: " + text

    return pipeline


# ============== 本地启动（可选） ==============
if __name__ == "__main__":
    pipeline = create_text2sql_pipeline(
        source="qwen",                # 规划 LLM 使用的在线模型源
        db_path=DB_PATH
    )
    WebModule(pipeline, port=range(23461, 23470)).start().wait()
````

</details>


