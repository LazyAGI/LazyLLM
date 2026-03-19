# 内置与扩展工具（lazyllm.tools）

以下工具均来自 **lazyllm.tools**，可供 Agent 直接或通过 `fc_register` 包装后调用。搜索类从 `lazyllm.tools.tools` 导入，其余从 `lazyllm.tools` 导入。

---

## 搜索类工具

### GoogleSearch

通过 Google Custom Search API 搜索指定关键词。

参数:

- custom_search_api_key (str) – 用户申请的 Google API key。
- search_engine_id (str) – 用户创建的搜索引擎 id。
- timeout (int, default: 10) – 请求超时时间（秒）。
- proxies (Dict[str, str], default: None) – 代理配置。

```python
from lazyllm.tools.tools import GoogleSearch

key = '<your_google_search_api_key>'
cx = '<your_search_engine_id>'
google = GoogleSearch(custom_search_api_key=key, search_engine_id=cx)
res = google(query='商汤科技', date_restrict='m1')
```

### TencentSearch

腾讯云内容搜索 API 封装，支持关键词搜索与结果处理。

参数:

- secret_id (str) – 腾讯云 API 密钥 ID。
- secret_key (str) – 腾讯云 API 密钥。

```python
from lazyllm.tools.tools import TencentSearch

searcher = TencentSearch(secret_id='<your_secret_id>', secret_key='<your_secret_key>')
res = searcher('calculus')
```

### 其他搜索工具

| 工具 | 说明 | 导入方式 |
|------|------|----------|
| BingSearch | Bing 搜索 | `from lazyllm.tools.tools import BingSearch` |
| BochaSearch | 博查搜索 API | `from lazyllm.tools.tools import BochaSearch` |
| WikipediaSearch | 维基百科检索 | `from lazyllm.tools.tools import WikipediaSearch` |
| ArxivSearch | arXiv 论文检索 | `from lazyllm.tools.tools import ArxivSearch` |
| StackOverflowSearch | Stack Overflow 检索 | `from lazyllm.tools.tools import StackOverflowSearch` |
| SemanticScholarSearch | Semantic Scholar 学术检索 | `from lazyllm.tools.tools import SemanticScholarSearch` |
| GoogleBooksSearch | Google 图书检索 | `from lazyllm.tools.tools import GoogleBooksSearch` |

---

## 通用工具

| 工具 | 说明 | 导入方式 |
|------|------|----------|
| HttpTool | 发起 HTTP 请求，可封装任意 REST API 供 Agent 调用 | `from lazyllm.tools import HttpTool` |
| Weather | 天气查询 | `from lazyllm.tools.tools import Weather` |
| Calculator | 数学计算 | `from lazyllm.tools.tools import Calculator` |
| JsonExtractor | 从文本中提取 JSON | `from lazyllm.tools.tools import JsonExtractor` |
| JsonConcentrator | 合并/聚合 JSON 结构 | `from lazyllm.tools.tools import JsonConcentrator` |

---

## SQL / 数据工具

### SqlManager

管理数据库连接与表结构，执行 SQL、获取表信息等，常与 SqlCall 配合实现自然语言转 SQL。

- 支持 SQLite、PostgreSQL、MySQL 等；远程库需填写 user、password、host、port。
- 常用方法: `get_all_tables`, `execute_query`, `execute_commit`, `create_table`, `set_desc` 等。

```python
from lazyllm.tools import SqlManager

sql_manager = SqlManager(db_path='/path/to/db.sqlite')  # 或指定 user/password/host/port
tables = sql_manager.get_all_tables()
result = sql_manager.execute_query('SELECT * FROM t LIMIT 10')
```

### SqlCall

将自然语言与表结构信息交给 LLM 生成 SQL，再通过 SqlManager 执行，适合与 ReactAgent 组合为 SQL Agent。

```python
from lazyllm.tools import SqlManager, SqlCall, fc_register, ReactAgent

sql_manager = SqlManager(db_path='/path/to/db.sqlite')
sql_call = SqlCall(llm, sql_manager, use_llm_for_sql_result=False)  # 可选参数见 API 文档

@fc_register('tool')
def query_db(query: str) -> str:
    return sql_call(query)

agent = ReactAgent(llm, tools=['query_db'])
agent('查询销售额最高的前 5 个产品')
```

更多示例见本目录 [basic.md](basic.md) 与 [agent.md](agent.md)，或项目主 docs 内 Cookbook（sql_agent、tabular 等）。

---

## 代码与能力类

### CodeGenerator

基于 LLM 生成代码，可与 FunctionCallAgent 等组合实现“自然语言 → 代码 → 执行”的代码 Agent。

```python
from lazyllm.tools import CodeGenerator, FunctionCallAgent, fc_register

gen = CodeGenerator(llm, prompt='根据用户需求生成可执行代码，只返回代码本身。')

@fc_register('tool')
def run_code(requirement: str) -> str:
    code = gen(requirement)
    # 可选：在沙箱中执行 code
    return code

agent = FunctionCallAgent(llm, tools=['run_code'])
agent('写一个函数计算斐波那契数列前 n 项')
```

### MCPClient

连接 MCP 服务器（本地或 SSE），获取 MCP 提供的工具列表，可直接作为 Agent 的 tools 使用。用法见 [基础组件 - MCPClient](../assets/agent/basic.md#4-mcpclient)。

---

## 小结

- **搜索与通用工具**: `lazyllm.tools.tools` 中的 Search、Weather、Calculator、Json*。
- **SQL / 数据**: `lazyllm.tools` 的 SqlManager、SqlCall（以及 MongoDBManager、DBManager 等）。
- **代码与 MCP**: `lazyllm.tools` 的 CodeGenerator、MCPClient；ToolManager、SkillManager 等见 Agent 模块文档。
- **自定义工具**: 使用 `fc_register('tool')` 注册函数，见 [基础组件](../assets/agent/basic.md)。

更多参数与用法见本页上文及 [references/agent.md](../../references/agent.md)。
