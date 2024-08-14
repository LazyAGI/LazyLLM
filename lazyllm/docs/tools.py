# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.tools)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.tools)
add_example = functools.partial(utils.add_example, module=lazyllm.tools)

add_english_doc('Document', '''\
Initializes a Document module with optional user interface creation.

This constructor initializes a Document module that can work with or without a user interface (UI). If the UI is enabled, it also sets up a server module to manage document operations and a web module for UI interactions. The module relies on an implementation (`DocGroupImpl`) that handles the core functionality such as generating signatures and querying documents based on signatures.

Args:
    dataset_path (str): The path to the dataset directory. This directory should contain the documents to be managed by the Document module.
    embed: An embedding object or function that is used for generating document embeddings. The exact type and requirements depend on the implementation of `DocGroupImpl`.
    create_ui (bool, optional): A flag indicating whether to create a user interface for the Document module. Defaults to True.
    launcher (optional): An object or function responsible for launching the server module. If not provided, a default launcher from `lazyllm.launchers` with asynchronous behavior (`sync=False`) is used.
''')

add_chinese_doc('Document', '''\
初始化一个具有可选用户界面的文档模块。

此构造函数初始化一个可以有或没有用户界面的文档模块。如果启用了用户界面，它还会提供一个ui界面来管理文档操作接口，并提供一个用于用户界面交互的网页。

Args:
    dataset_path (str): 数据集目录的路径。此目录应包含要由文档模块管理的文档。
    embed: 用于生成文档embedding的对象。
    create_ui (bool, optional): 指示是否为文档模块创建用户界面的标志。默认为 True。
    launcher (optional): 负责启动服务器模块的对象或函数。如果未提供，则使用 `lazyllm.launchers` 中的默认异步启动器 (`sync=False`)。
''')

add_example('Document', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.docment import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
''')

add_english_doc('Reranker', '''\
Initializes a Rerank module for postprocessing and reranking of nodes (documents).
This constructor initializes a Reranker module that configures a reranking process based on a specified reranking type. It allows for the dynamic selection and instantiation of reranking kernels (algorithms) based on the type and provided keyword arguments.

Args:
    types: The type of reranker to be used for the postprocessing and reranking process. Defaults to 'Reranker'.
    **kwargs: Additional keyword arguments that are passed to the reranker upon its instantiation.

**Detailed explanation of reranker types**\n
- Reranker：This registered reranking function instantiates a SentenceTransformerRerank reranker with a specified model and top_n parameter. It is designed to rerank nodes based on sentence transformer embeddings.\n
- SimilarityFilter：This registered reranking function instantiates a SimilarityPostprocessor with a specified similarity threshold. It filters out nodes that do not meet the similarity criteria.\n
- KeywordFilter：This registered reranking function instantiates a KeywordNodePostprocessor with specified required and excluded keywords. It filters nodes based on the presence or absence of these keywords.
''')

add_chinese_doc('Reranker', '''\
用于创建节点（文档）后处理和重排序的模块。

Args:
    types: 用于后处理和重排序过程的排序器类型。默认为 'Reranker'。
    **kwargs: 传递给重新排序器实例化的其他关键字参数。

详细解释types类型
  - Reranker：实例化一个具有指定模型和 top_n 参数的 SentenceTransformerRerank 重排序器。
  - SimilarityFilter：实例化一个具有指定相似度阈值的 SimilarityPostprocessor。它过滤掉不满足相似度标准的节点。
  - KeywordFilter：实例化一个具有指定必需和排除关键字的 KeywordNodePostprocessor。它根据这些关键字的存在或缺失来过滤节点。
''')

add_example('Reranker', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.base import Reranker, Retriever
>>> from lazyllm.tools.rag.docment import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> rm = Retriever(documents, similarity='chinese_bm25', parser='SentenceDivider', similarity_top_k=6)
>>> reranker = Reranker(types='SimilarityFilter', threshold=2.0)
>>> m = lazyllm.ActionModule(rm, reranker)
>>> m.start()
>>> print(m("query"))  
''')

add_english_doc('Retriever', '''\
Create a Retriever module for document query and retrieval.
This constructor initializes a Retriever module that configures a document retrieval process based on a specified similarity measure. It generates a signature using the document's generate_signature method, which is then used for querying documents.

Args:
    doc: The Document module instance that contains the documents and functionalities for generating signatures and querying documents.
    parser: The parser to be used for processing documents during querying.The candidate sets are ["Hierarchy", "CoarseChunk", "MediumChunk", "FineChunk", "SentenceDivider", "TokenDivider", "HtmlExtractor", "JsonExtractor", "MarkDownExtractor"]
    similarity : The name of the similarity function to be used for document retrieval. Defaults to 'defatult'. The candidate sets are ["defatult", "chinese_bm25", "bm25"].
    index: The type of index to be used for document retrieval. Defaults to 'vector'.

**Detailed explanation of Parser types**\n
  - Hierarchy: This node parser will chunk nodes into hierarchical nodes. This means a single input will be chunked into several hierarchies of chunk sizes, with each node containing a reference to it's parent node.\n
  - CoarseChunk: Get children of root nodes in given nodes that have given depth 0.\n
  - MediumChunk: Get children of root nodes in given nodes that have given depth 1.\n
  - FineChunk: Get children of root nodes in given nodes that have given depth 2.\n
  - SentenceDivider：This node parser attempts to split text while respecting the boundaries of sentences.\n
  - TokenDivider：This node parser attempts to split to a consistent chunk size according to raw token counts.\n
  - HtmlExtractor：This node parser uses beautifulsoup to parse raw HTML.By default, it will parse a select subset of HTML tags, but you can override this.The default tags are: ["p", "h1"]\n
  - JsonExtractor：Splits a document into Nodes using custom JSON splitting logic.\n
  - MarkDownExtractor：Splits a document into Nodes using custom Markdown splitting logic.

**Detailed explanation of similarity types**\n
  - defatult：Search documents using cosine method.\n
  - chinese_bm25：Search documents using chinese_bm25 method.The primary differences between chinese_bm25 and the standard bm25 stem from the specific adjustments and optimizations made for handling Chinese text. \n
  - bm25：Search documents using bm25 method.\n
''')

add_chinese_doc('Retriever', '''\
创建一个用于文档查询和检索的检索模块。

此构造函数初始化一个检索模块，该模块根据指定的相似度度量配置文档检索过程。它使用文档的 generate_signature 方法生成签名，然后用于查询文档。

Args:
    doc: 文档模块实例。
    parser: 用于设置处理文档的解析器。候选集包括 ["Hierarchy", "CoarseChunk", "MediumChunk", "FineChunk", "SentenceDivider", "TokenDivider", "HtmlExtractor", "JsonExtractor", "MarkDownExtractor"]
    similarity : 用于设置文档检索的相似度函数。默认为 'defatult'。候选集包括 ["defatult", "chinese_bm25", "bm25"]。
    index: 用于文档检索的索引类型。默认为 'vector'。

详细解释parser类型\n
  - Hierarchy: 此节点解析器将原始文本按照分层算法拆分成节点。这意味着一个输入将被分块成几个层次的块，每个节点包含对其父节点的引用。\n
  - CoarseChunk: 获取给定结点中的深度为 0 的子节点。\n
  - MediumChunk: 获取给定结点中的深度为 1 的子节点。\n
  - FineChunk: 获取给定结点中的深度为 2 的子节点。\n
  - SentenceDivider：此节点解析器尝试在尊重句子边界的同时拆分文本。\n
  - TokenDivider：此节点解析器尝试将原始文本拆分为一致的块大小。\n
  - HtmlExtractor：此节点解析器使用 beautifulsoup 解析原始 HTML。默认情况下，它将解析选定的 HTML 标签，但您可以覆盖此设置。默认标签为: ["p", "h1"]\n
  - JsonExtractor：使用自定义 JSON 拆分逻辑将文档拆分为节点。\n
  - MarkDownExtractor：使用自定义 Markdown 拆分逻辑将文档拆分为节点。\n

详细解释similarity类型\n
  - defatult：使用 cosine 算法搜索文档。\n
  - chinese_bm25：使用 chinese_bm25 算法搜索文档。chinese_bm25 与标准 bm25 的主要区别在于对中文的特定优化。\n
  - bm25：使用 bm25 算法搜索文档。\n
''')

add_example('Retriever', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.base import Retriever
>>> from lazyllm.tools.rag.docment import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> rm = Retriever(documents, similarity='chinese_bm25', parser='SentenceDivider', similarity_top_k=6)
>>> rm.start()
>>> print(rm("query"))
''')

add_chinese_doc('WebModule', '''\
WebModule是LazyLLM为开发者提供的基于Web的交互界面。在初始化并启动一个WebModule之后，开发者可以从页面上看到WebModule背后的模块结构，并将Chatbot组件的输入传输给自己开发的模块进行处理。
模块返回的结果和日志会直接显示在网页的“处理日志”和Chatbot组件上。除此之外，WebModule支持在网页上动态加入Checkbox或Text组件用于向模块发送额外的参数。
WebModule页面还提供“使用上下文”，“流式输出”和“追加输出”的Checkbox，可以用来改变页面和后台模块的交互方式。

<span style="font-size: 20px;">&ensp;**`WebModule.init_web(component_descs) -> gradio.Blocks`**</span>
使用gradio库生成演示web页面，初始化session相关数据以便在不同的页面保存各自的对话和日志，然后使用传入的component_descs参数为页面动态添加Checkbox和Text组件，最后设置页面上的按钮和文本框的相应函数
之后返回整个页面。WebModule的__init__函数调用此方法生成页面。

Args：
    component_descs (list): 用于动态向页面添加组件的列表。列表中的每个元素也是一个列表，其中包含5个元素，分别是组件对应的模块ID，模块名，组件名，组件类型（目前仅支持Checkbox和Text），组件默认值。
''')

add_english_doc('WebModule', '''\
WebModule is a web-based interactive interface provided by LazyLLM for developers. After initializing and starting
a WebModule, developers can see structure of the module they provides behind the WebModule, and transmit the input
of the Chatbot component to their modules. The results and logs returned by the module will be displayed on the 
“Processing Logs” and Chatbot component on the web page. In addition, Checkbox or Text components can be added
programmatically to the web page for additional parameters to the background module. Meanwhile, The WebModule page
provides Checkboxes of “Use Context,” “Stream Output,” and “Append Output,” which can be used to adjust the
interaction between the page and the module behind.

<span style="font-size: 20px;">&ensp;**`WebModule.init_web(component_descs) -> gradio.Blocks`**</span>

Generate a demonstration web page based on gradio. The function initializes session-related data to save chat history
and logs for different pages, then dynamically add Checkbox and Text components to the page according to component_descs
parameter, and set the corresponding functions for the buttons and text boxes on the page at last.
WebModule’s __init__ function calls this method to generate the page.

Args:
    component_descs (list): A list used to add components to the page. Each element in the list is also a list containing 
    5 elements, which are the module ID, the module name, the component name, the component type (currently only
    supports Checkbox and Text), and the default value of the component.

''')

add_example('WebModule', '''\
>>> import lazyllm
>>> def func2(in_str, do_sample=True, temperature=0.0, *args, **kwargs):
...     return f"func2:{in_str}|do_sample:{str(do_sample)}|temp:{temperature}"
...
>>> m1=lazyllm.ActionModule(func2)
>>> m1.name="Module1"
>>> w = lazyllm.WebModule(m1, port=[20570, 20571, 20572], components={
...         m1:[('do_sample', 'Checkbox', True), ('temperature', 'Text', 0.1)]},
...                       text_mode=lazyllm.tools.WebModule.Mode.Refresh)
>>> w.start()
193703: 2024-06-07 10:26:00 lazyllm SUCCESS: ...
''')

add_chinese_doc(
    "SQLiteTool",
    """\
SQLiteTool是与SQLite数据库进行交互的专用工具。它扩展了SqlTool类，提供了创建表、执行查询和对SQLite数据库进行更新的方法。

Arguments:
    db_file (str): SQLite 文件数据库的路径
""",
)

add_english_doc(
    "SQLiteTool",
    """\
SQLiteTool is a specialized tool for interacting with SQLite databases.
It extends the SqlTool class and provides methods for creating tables, executing queries, and performing updates on SQLite databases.


Arguments:
    db_file (str): The path to the SQLite database file.
""",
)

add_example(
    "SQLiteTool",
    """\
    >>> from lazyllm.tools import SQLiteTool
    >>> sql_tool = SQLiteTool("personal.db")
    >>> tables_info = {
    ...     "User": {
    ...         "fields": {
    ...             "id": {
    ...                 "type": "integer",
    ...                 "comment": "user id"
    ...             },
    ...             "name": {
    ...                 "type": "text",
    ...                 "comment": "user name"
    ...             },
    ...             "email": {
    ...                 "type": "text",
    ...                 "comment": "user email"
    ...             }
    ...         }
    ...     }
    ... }
    >>> sql_tool.create_tables(tables_info)
    >>> sql_tool.sql_update("INSERT INTO User (id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
    >>> table_info = sql_tool.get_all_tables()
    >>> print(table_info)
    >>> result_json = sql_tool.get_query_result_in_json("SELECT * from User")
    >>> print(result_json)
""",
)

add_chinese_doc(
    "SQLiteTool.create_tables",
    """\
根据描述表结构的JSON字典在SQLite数据库中创建表。
JSON格式形如：{$TABLE_NAME:{"fields":{$COLUMN_NAME:{"type":("REAL"/"TEXT"/"INT"), "comment":"..."} } } }
""",
)

add_english_doc(
    "SQLiteTool.create_tables",
    """\
Create tables According to tables json dict.
THis JSON format should be as：{$TABLE_NAME:{"fields":{$COLUMN_NAME:{"type":("REAL"/"TEXT"/"INT"), "comment":"..."} } } }
""",
)

add_example(
    "SQLiteTool.create_tables",
    """\
>>> from lazyllm.tools import SQLiteTool
>>> sql_tool = SQLiteTool("personal.db")
>>> tables_info = {
...     "User": {
...         "fields": {
...             "id": {
...                 "type": "integer",
...                 "comment": "user id"
...             },
...             "name": {
...                 "type": "text",
...                 "comment": "user name"
...             },
...             "email": {
...                 "type": "text",
...                 "comment": "user email"
...             }
...         }
...     }
... }
>>> sql_tool.create_tables(tables_info)
""",
)

add_chinese_doc(
    "SQLiteTool.get_all_tables",
    """\
检索并返回SQLite数据库中所有表的字符串表示形式。
""",
)

add_english_doc(
    "SQLiteTool.get_all_tables",
    """\
Retrieves and returns a string representation of all the tables in the SQLite database.
""",
)

add_example(
    "SQLiteTool.get_all_tables",
    """\
>>> from lazyllm.tools import SQLiteTool
>>> sql_tool = SQLiteTool("personal.db")
>>> tables_info = sql_tool.get_all_tables()
>>> print(tables_info)
CREATE TABLE employee
(
    employee_id INT comment '工号',
    first_name TEXT comment '姓',
    last_name TEXT comment '名',
    department TEXT comment '部门'
)
CREATE TABLE sales
(
    employee_id INT comment '工号',
    q1_2023 REAL comment '2023年第1季度销售额',
    q2_2023 REAL comment '2023年第2季度销售额',
    q3_2023 REAL comment '2023年第3季度销售额',
    q4_2023 REAL comment '2023年第4季度销售额'
)
""",
)

add_chinese_doc(
    "SQLiteTool.get_query_result_in_json",
    """\
执行SQL查询并返回JSON格式的结果。
""",
)

add_english_doc(
    "SQLiteTool.get_query_result_in_json",
    """\
Executes a SQL query and returns the result in JSON format.
""",
)

add_example(
    "SQLiteTool.get_query_result_in_json",
    """\
>>> from lazyllm.tools import SQLiteTool
>>> sql_tool = SQLiteTool("personal.db")
>>> result_json = sql_tool.get_query_result_in_json("SELECT * from sales limit 1")
>>> print(result_json)
[{employee_id: 8, q1_2023: 3471.41, q2_2023: 14789.25, q3_2023: 3478.34, q4_2023: 1254.23}]
""",
)

add_chinese_doc(
    "SQLiteTool.sql_update",
    """\
在SQLite数据库上执行SQL插入或更新脚本。
""",
)

add_english_doc(
    "SQLiteTool.sql_update",
    """\
Execute insert or update script.
""",
)

add_example(
    "SQLiteTool.sql_update",
    """\
>>> from lazyllm.tools import SQLiteTool
>>> sql_tool = SQLiteTool("personal.db")
>>> sql_tool.sql_update("INSERT INTO sales VALUES (1, 8715.55, 8465.65, 24747.82, 3514.36);")
""",
)

add_chinese_doc(
    "IntentClassifier",
    """\
IntentClassifier 是一个基于语言模型的意图识别器，用于根据用户提供的输入文本及对话上下文识别预定义的意图，并通过预处理和后处理步骤确保准确识别意图。

Arguments:
    llm: 用于意图识别的语言模型对象，OnlineChatModule或TrainableModule类型
    intent_list (list)：包含所有可能意图的字符串列表。可以包含中文或英文的意图。
    return_trace (bool, 可选)：如果设置为 True，则将结果记录在trace中。默认为 False。
""",
)

add_english_doc(
    "IntentClassifier",
    """\
IntentClassifier is an intent recognizer based on a language model that identifies predefined intents based on user-provided input text and conversational context.
It can handle intent lists and ensures accurate intent recognition through preprocessing and postprocessing steps.

Arguments:
    llm: A language model object used for intent recognition, which can be of type OnlineChatModule or TrainableModule.
    intent_list (list): A list of strings containing all possible intents. This list can include intents in either Chinese or English.
    return_trace (bool, optional): If set to True, the results will be recorded in the trace. Defaults to False.
""",
)

add_example(
    "IntentClassifier",
    """\
    >>> import lazyllm
    >>> classifier_llm = lazyllm.OnlineChatModule(model=MODEL_ID, source="openai", base_url=BASE_URL)
    >>> chatflow_intent_list = ["闲聊", "金融知识问答", "销售业绩查询", "员工信息查询"]
    >>> classifier = IntentClassifier(classifier_llm, intent_list=chatflow_intent_list)
    >>> classifier.start()
    >>> print(classifier(QUERY))  
""",
)

add_chinese_doc(
    "SqlModule",
    """\
SqlModule 是一个扩展自 ModuleBase 的类,提供了使用语言模型(LLM)生成和执行 SQL 查询的接口。
它设计用于与 SQL 数据库交互,从语言模型的响应中提取 SQL 查询,执行这些查询,并返回结果或解释。

Arguments:
    llm: 用于生成和解释 SQL 查询及解释的大语言模型。
    sql_tool (SqlTool)：一个 SqlTool 实例，用于处理与 SQL 数据库的交互。
    output_in_json (bool, 可选): 如果设置为True,管道只会输出原始的SQL结果而不进行进一步处理。默认值为False。
    return_trace (bool, 可选): 如果设置为 True,则将结果记录在trace中。默认为 False。
""",
)

add_english_doc(
    "SqlModule",
    """\
SqlModule is a class that extends ModuleBase and provides an interface for generating and executing SQL queries using a language model (LLM). 
It is designed to interact with a SQL database, extract SQL queries from LLM responses, execute those queries, and return results or explanations.

Arguments:
    llm: A language model to be used for generating and interpreting SQL queries and explanations.
    sql_tool (SqlTool): An instance of SqlTool that handles interaction with the SQL database.
    output_in_json (bool, optional): If set to True, the pipeline will only output raw SQL results without further processing. Default is False.
    return_trace (bool, optional): If set to True, the results will be recorded in the trace. Defaults to False.  
""",
)

add_example(
    "SqlModule",
    """\
    >>> import lazyllm
    >>> from lazyllm.tools import SQLiteTool, SqlModule
    >>> sql_tool = SQLiteTool("personal.db")
    >>> sql_llm = lazyllm.OnlineChatModule(model="gpt-4o", source="openai", base_url="***")
    >>> sql_module = SqlModule(sql_llm, sql_tool, output_in_json=False)
    >>> print(sql_module("员工Alice的邮箱地址是什么?"))
""",
)
