# 多源数据加载与智能 SQL 查询处理

## 本项目展示了如何通过 Python 实现多格式文件加载（CSV/Excel）与自然语言驱动的 SQL 查询，支持数据自动预览和多格式解析。

## !!! abstract "核心功能"
- **多格式文件读取**：支持 CSV、XLSX 文件自动解析
- **灵活编码检测**：通过 `chardet` 自动识别文件编码
- **智能 SQL 查询**：基于自然语言自动生成 SQL 并查询 SQLite 数据库
- **数据可视化预览**：在控制台输出原始数据

---

## 环境准备

需安装以下依赖包：

```bash
pip install lazyllm chardet pandas sqlite3
```

---

## 结构解析
### 1. 文件编码检测

在加载 CSV 文件前，通过 `chardet` 检测其编码，避免读取时出现乱码问题：

```python
with open(csv_path, 'rb') as f:
    print(chardet.detect(f.read()))
```

---

### 2. 多格式文件加载

#### **第一种方式**：直接使用 `SimpleDirectoryReader` 读取多种文件格式

```python
loader1 = SimpleDirectoryReader(
    input_files=[csv_path, xlsx_path],
    exclude_hidden=True,
    recursive=False
)
documents = loader1.forward()
for doc in documents:
    print(doc.text)
```
- **特点**：自动解析多种文件格式，支持批量读取。
- **用途**：快速获取原始数据内容，便于后续处理。

#### **第二种方式**：自定义文件解析和加载

```python
loader2 = SimpleDirectoryReader(
    input_files=[csv_path],
    recursive=True,
    exclude_hidden=True,
    num_files_limit=10,
    required_exts=[".csv"],
    file_extractor={
        "*.csv": PandasCSVReader(
            concat_rows=False,
            col_joiner=" | ",
            row_joiner="\n\n",
            pandas_config={"sep": None, "engine": "python", "header": None},
        )
    }
)
documents2 = loader2.forward()
for docs in documents2:
    print(docs.text)
```
- **特点**：
  - 使用 对用文件格式`Reader` 自定义列、行连接方式
  - 支持无表头数据（`header=None`）
- **用途**：更灵活地处理数据格式

---

### 3. SQLite 数据库管理

使用 `SqlManager` 连接 SQLite 数据库并查看表结构与数据：

```python
sql_manager = SqlManager(
    db_type="sqlite",
    user="",
    password="",
    host="",
    port=0,
    db_name=sql_path
)

print(sql_manager.desc)  # 数据库模式描述
result_json = sql_manager.execute_query("SELECT * FROM employees;")
```

---
### 1. File Encoding Detection

Before loading a CSV file, use chardet to detect its encoding to avoid garbled text issues during reading.：

```python
with open(csv_path, 'rb') as f:
    print(chardet.detect(f.read()))
```

---

### 2. Multi-Format File Loading

#### **Method 1:**：Use SimpleDirectoryReader directly to read multiple file formats

```python
loader1 = SimpleDirectoryReader(
    input_files=[csv_path, xlsx_path],
    exclude_hidden=True,
    recursive=False
)
documents = loader1.forward()
for doc in documents:
    print(doc.text)
```
- **Features**：Automatically parses multiple file formats and supports batch reading.
- **Use Case**：Quickly obtain raw data content for subsequent processing.

 

#### **Method 2**：Custom File Parsing and Loading

```python
loader2 = SimpleDirectoryReader(
    input_files=[csv_path],
    recursive=True,
    exclude_hidden=True,
    num_files_limit=10,
    required_exts=[".csv"],
    file_extractor={
        "*.csv": PandasCSVReader(
            concat_rows=False,
            col_joiner=" | ",
            row_joiner="\n\n",
            pandas_config={"sep": None, "engine": "python", "header": None},
        )
    }
)
documents2 = loader2.forward()
for docs in documents2:
    print(docs.text)
```
- **Features**：
  - Use the corresponding file format Reader to customize column and row concatenation methods
  - Supports data without headers (header=None)）
- **Use Case**：Provides more flexibility in handling data formats

---

### 3. SQLite Database Management

Use SqlManager to connect to an SQLite database, view table structures, and inspect data.

```python
sql_manager = SqlManager(
    db_type="sqlite",
    user="",
    password="",
    host="",
    port=0,
    db_name=sql_path
)

print(sql_manager.desc)  
result_json = sql_manager.execute_query("SELECT * FROM employees;")
```

---

### 4. 自然语言转 SQL 查询

通过 `lazyllm` 的 `SqlCall` 模块，实现自然语言到 SQL 的自动转换与执行：

```python
llm = lazyllm.OnlineChatModule()

sql_manager = SqlManager(
    db_type="sqlite",
    user="",
    password="",
    host="",
    port=0,
    db_name=sql_path
)

print("=== Schema Description ===")
print(sql_manager.desc)
print("=== Current employees table data ===")
result_json = sql_manager.execute_query("SELECT * FROM employees;")
try:
    result = json.loads(result_json)
    for row in result:
        print(row)
except json.JSONDecodeError:
    print(result_json)
llm = lazyllm.OnlineChatModule()

sql_examples = """SELECT COUNT(*) AS cnt FROM employees;"""
sql_call = SqlCall(
    llm=llm,
    sql_manager=sql_manager,
    sql_examples=sql_examples,
    use_llm_for_sql_result=True,
    return_trace=False
)

question = "How many employees are there in the company?"
answer = sql_call.forward(question)

print("Question:", question)
print("Answer:", answer)

```
- **流程**：
  1. 用户输入自然语言问题
  2. 模型将其转换为 SQL 语句
  3. 执行查询并返回结果

---
### 5. 结果输出
````
=== Data ===
1, 张三, 28.0, 75000.5, 销售部, 2020-03-15, 沟通,谈判, True, 优秀员工
2, 李四, 32.0, 92500.0, 技术部, 2018-07-22, Python;Java, True, 项目负责人
3, 王五, 45.0, 120000.0, 管理层, 2015-11-30, 领导力;战略规划, True, 总监
4, 赵六, 23.0, 65000.0, 市场部, 2023-01-10, 社交媒体, False, 试用期
5, 钱七, 29.0, 88000.75, 人力资源部, 2021-09-05, 招聘;培训, True, nan
6, 孙八, nan, 99999.99, 财务部, 2022-05-20, 会计;审计, True, 新晋升
7, 周九, 31.0, 0.0, 待分配, 2023-03-01, nan, False, 离职状态
8, 吴十, 28.0, 55000.0, 客户支持, 2020-12-15, 英语,日语, True, 多语言支持
9, 郑十一, 35.0, 110000.0, 技术部, 2019-04-18, AI;机器学习, True, 首席工程师
10, 王十二, 27.0, 72000.5, 销售部, 2021-11-30, 客户关系;CRM, True, 年度最佳销售
=== Data ===
1 Alice Chen 28 2020-03-15 $75,000 Marketing SEO, Analytics True
2 Bob Smith 32 2018-07-22 $92,500 Engineering Python, Java True
3 Carol Wang 45 2015-11-30 $120,000 Management Leadership, Strategy True
4 Dave Brown 23 2023-01-10 $65,000 Sales Negotiation False
5 Eve Davis 29 2021-09-05 $88,000 HR Recruitment, Training True
ID | Full Name | Age | Salary | Department | Join Date | Skills | Is Active | Notes
001 | 张三 | 28 | 75000.5 | 销售部 | 2020-03-15 | 沟通,谈判 | TRUE | 优秀员工
002 | 李四 | 32 | 92500.0 | 技术部 | 2018-07-22 | Python;Java | TRUE | 项目负责人
003 | 王五 | 45 | 120000.0 | 管理层 | 2015-11-30 | 领导力;战略规划 | TRUE | 总监
004 | 赵六 | 23 | 65000.0 | 市场部 | 2023-01-10 | 社交媒体 | FALSE | 试用期
005 | 钱七 | 29 | 88000.75 | 人力资源部 | 2021-09-05 | 招聘;培训 | TRUE | nan
006 | 孙八 | nan | 99999.99 | 财务部 | 2022-05-20 | 会计;审计 | TRUE | 新晋升
007 | 周九 | 31 | 0.0 | 待分配 | 2023-03-01 | nan | FALSE | 离职状态
008 | 吴十 | 28 | 55000.0 | 客户支持 | 2020-12-15 | 英语,日语 | TRUE | 多语言支持
009 | 郑十一 | 35 | 110000.0 | 技术部 | 2019-04-18 | AI;机器学习 | TRUE | 首席工程师
010 | 王十二 | 27 | 72000.5 | 销售部 | 2021-11-30 | 客户关系;CRM | TRUE | 年度最佳销售
=== Schema Description ===
The tables description is as follows
```
Table employees
(
 EmployeeId INTEGER,
 FirstName TEXT,
 LastName TEXT,
 Title TEXT
);
```

=== Current employees table data ===
{'EmployeeId': 1, 'FirstName': 'John', 'LastName': 'Doe', 'Title': 'Software Engineer'}
{'EmployeeId': 2, 'FirstName': 'Jane', 'LastName': 'Smith', 'Title': 'Data Analyst'}
{'EmployeeId': 3, 'FirstName': 'Alice', 'LastName': 'Johnson', 'Title': 'Project Manager'}
{'EmployeeId': 4, 'FirstName': 'Bob', 'LastName': 'Lee', 'Title': 'QA Engineer'}
Question: How many employees are there in the company?
Answer: Based on the execution results, there are 4 employees in the company.
```