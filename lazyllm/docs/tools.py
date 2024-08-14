# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.tools)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.tools)
add_example = functools.partial(utils.add_example, module=lazyllm.tools)

# ---------------------------------------------------------------------------- #

# rag/document.py

add_english_doc('Document', '''\
Initialize a document module with an optional user interface.

This constructor initializes a document module that can have an optional user interface. If the user interface is enabled, it also provides a UI to manage the document operation interface and offers a web page for user interface interaction.

Args:

- dataset_path (str): The path to the dataset directory. This directory should contain the documents to be managed by the document module.
- embed: The object used to generate document embeddings.
- create_ui (bool, optional): A flag indicating whether to create a user interface for the document module. Defaults to True.
- launcher (optional): An object or function responsible for launching the server module. If not provided, the default asynchronous launcher from `lazyllm.launchers` is used (`sync=False`).
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
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
''')

add_english_doc('Document.create_node_group', '''
Generate a node group produced by the specified rule.

Args:

- name (str): The name of the node group.
- transform (Callable): The transformation rule that converts a node into a node group. The function prototype is `(DocNode, group_name, **kwargs) -> List[DocNode]`. Currently built-in options include [SentenceSplitter][lazyllm.tools.SentenceSplitter], and users can define their own transformation rules.
- parent (str): The node that needs further transformation. The series of new nodes obtained after transformation will be child nodes of this parent node. If not specified, the transformation starts from the root node.
- kwargs: Parameters related to the specific implementation.
''')

add_chinese_doc('Document.create_node_group', '''
创建一个由指定规则生成的 node group。

Args:
    name (str)：node group 的名称。
    transform (Callable)：将 node 转换成 node group 的转换规则，函数原型是 `(DocNode, group_name, **kwargs) -> List[DocNode]`。目前内置的有 [SentenceSplitter][lazyllm.tools.SentenceSplitter]。用户也可以自定义转换规则。
    parent (str)：需要进一步转换的节点。转换之后得到的一系列新的节点将会作为该父节点的子节点。如果不指定则从根节点开始转换。
    kwargs：和具体实现相关的参数。
''')

add_example('Document.create_node_group', '''
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
''')

add_english_doc('Document.find_parent', '''
Find the parent node of the specified node.

Args:

- group (str): The name of the node for which to find the parent.
''')

add_chinese_doc('Document.find_parent', '''
查找指定节点的父节点。

Args:
    group (str)：需要查找的节点名称
''')

add_example('Document.find_parent', '''
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
>>> documents.find_parent('children')
''')

add_english_doc('Document.find_children', '''
Find the child nodes of the specified node.

Args:

- group (str): The name of the node for which to find the children.
''')

add_chinese_doc('Document.find_children', '''
查找指定节点的子节点。

Args:
    group (str)：需要查找的名称
''')

add_example('Document.find_children', '''
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
>>> documents.find_children('parent')
''')

# ---------------------------------------------------------------------------- #

# rag/rerank.py

add_english_doc('Reranker', '''\
Initializes a Rerank module for postprocessing and reranking of nodes (documents).
This constructor initializes a Reranker module that configures a reranking process based on a specified reranking type. It allows for the dynamic selection and instantiation of reranking kernels (algorithms) based on the type and provided keyword arguments.

Args:

- name: The type of reranker to be used for the postprocessing and reranking process. Defaults to 'Reranker'.
- kwargs: Additional keyword arguments that are passed to the reranker upon its instantiation.

**Detailed explanation of reranker types**

- Reranker: This registered reranking function instantiates a SentenceTransformerRerank reranker with a specified model and top_n parameter. It is designed to rerank nodes based on sentence transformer embeddings.\n
- KeywordFilter: This registered reranking function instantiates a KeywordNodePostprocessor with specified required and excluded keywords. It filters nodes based on the presence or absence of these keywords.
''')

add_chinese_doc('Reranker', '''\
用于创建节点（文档）后处理和重排序的模块。

Args:
    name: 用于后处理和重排序过程的排序器类型。默认为 'Reranker'。
    **kwargs: 传递给重新排序器实例化的其他关键字参数。

详细解释排序器类型

  - Reranker：实例化一个具有指定模型和 top_n 参数的 SentenceTransformerRerank 重排序器。
  - KeywordFilter：实例化一个具有指定必需和排除关键字的 KeywordNodePostprocessor。它根据这些关键字的存在或缺失来过滤节点。
''')

add_example('Reranker', '''
>>> import lazyllm
>>> from lazyllm.tools import Reranker, Retriever
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> rm = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off='0.01', topk=6)
>>> reranker = Reranker(name='ModuleReranker')
>>> m = lazyllm.ActionModule(rm, reranker)
>>> m.start()
>>> print(m("query"))
''')

# ---------------------------------------------------------------------------- #

# rag/retriever.py

add_english_doc('Retriever', '''
Create a retrieval module for document querying and retrieval. This constructor initializes a retrieval module that configures the document retrieval process based on the specified similarity metric.

Args:

- doc: An instance of the document module.
- group_name: The name of the node group on which to perform the retrieval.
- similarity: The similarity function to use for setting up document retrieval. Defaults to 'dummy'. Candidates include ["bm25", "bm25_chinese", "cosine"].
- similarity_cut_off: Discard the document when the similarity is below the specified value.
- index: The type of index to use for document retrieval. Currently, only 'default' is supported.
- topk: The number of documents to retrieve with the highest similarity.
- similarity_kw: Additional parameters to pass to the similarity calculation function.

The `group_name` has three built-in splitting strategies, all of which use `SentenceSplitter` for splitting, with the difference being in the chunk size:

- CoarseChunk: Chunk size is 1024, with an overlap length of 100
- MediumChunk: Chunk size is 256, with an overlap length of 25
- FineChunk: Chunk size is 128, with an overlap length of 12
''')

add_chinese_doc('Retriever', '''
创建一个用于文档查询和检索的检索模块。此构造函数初始化一个检索模块，该模块根据指定的相似度度量配置文档检索过程。

Args:
    doc: 文档模块实例。
    group_name: 在哪个 node group 上进行检索。
    similarity: 用于设置文档检索的相似度函数。默认为 'dummy'。候选集包括 ["bm25", "bm25_chinese", "cosine"]。
    similarity_cut_off: 当相似度低于指定值时丢弃该文档。
    index: 用于文档检索的索引类型。目前仅支持 'default'。
    topk: 表示取相似度最高的多少篇文档。
    similarity_kw: 传递给 similarity 计算函数的其它参数。

其中 `group_name` 有三个内置的切分策略，都是使用 `SentenceSplitter` 做切分，区别在于块大小不同：

- CoarseChunk: 块大小为 1024，重合长度为 100
- MediumChunk: 块大小为 256，重合长度为 25
- FineChunk: 块大小为 128，重合长度为 12
''')

add_example('Retriever', '''
>>> import lazyllm
>>> from lazyllm.tools import Retriever
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> rm = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off='0.01', topk=6)
>>> rm.start()
>>> print(rm("query"))
''')

# ---------------------------------------------------------------------------- #

# rag/transform.py

add_english_doc('SentenceSplitter', '''
Split sentences into chunks of a specified size. You can specify the size of the overlap between adjacent chunks.

Args:

- chunk_size (int): The size of the chunk after splitting.
- chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
''')

add_chinese_doc('SentenceSplitter', '''
将句子拆分成指定大小的块。可以指定相邻块之间重合部分的大小。

Args：
    chunk_size (int)：拆分之后的块大小
    chunk_overlap (int)；相邻两个块之间重合的内容长度
''')

add_example('SentenceSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
''')

add_english_doc('LLMParser', '''
A text summarizer and keyword extractor that is responsible for analyzing the text input by the user and providing concise summaries or extracting relevant keywords based on the requested task.

Args:

- llm (TrainableModule): A trainable module.
- language (str): The language type, currently only supports Chinese (zh) and English (en).
- task_type (str): Currently supports two types of tasks: summary and keyword extraction.
''')

add_chinese_doc('LLMParser', '''
一个文本摘要和关键词提取器，负责分析用户输入的文本，并根据请求任务提供简洁的摘要或提取相关关键词。

Args：
    llm (TrainableModule)：可训练的模块
    language (str)：语言种类，目前只支持中文（zh）和英文（en）
    task_type (str)：目前支持两种任务：摘要（summary）和关键词抽取（keywords）。
''')

add_example('LLMParser', '''
>>> import lazyllm
>>> from lazyllm.tools import LLMParser, TrainableModule
>>> llm = TrainableModule("internlm2-chat-7b")
>>> summary_parser = LLMParser(llm, language="en", task_type="summary")
''')

add_english_doc('LLMParser.transform', '''
Perform the set task on the specified document.

Args:

- node (DocNode): The document on which the extraction task needs to be performed.
''')

add_chinese_doc('LLMParser.transform', '''
在指定的文档上执行设定的任务。

Args：
    node (DocNode)：需要执行抽取任务的文档。
''')

add_example('LLMParser.transform', '''
>>> import lazyllm
>>> from lazyllm.tools import LLMParser, TrainableModule
>>> llm = TrainableModule("internlm2-chat-7b")
>>> summary_parser = LLMParser(llm, language="en", task_type="summary")
>>> keywords_parser = LLMParser(llm, language="en", task_type="keywords")
>>> documents = Document(dataset_path='your_doc_path', embed=m, create_ui=False)
>>> rm = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off='0.01', topk=6)
>>> summary_result = summary_parser.transform(rm[0])
>>> keywords_result = keywords_parser.transform(rm[0])
''')

# ---------------------------------------------------------------------------- #

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
    use_llm_for_sql_result (bool, 可选): 默认值为True。如果设置为False, 则只输出JSON格式表示的sql执行结果；True则会使用LLM对sql执行结果进行解读并返回自然语言结果。
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
    use_llm_for_sql_result (bool, optional): Default is True. If set to False, the module will only output raw SQL results in JSON without further processing.
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
    >>> sql_module = SqlModule(sql_llm, sql_tool, use_llm_for_sql_result=True)
    >>> print(sql_module("员工Alice的邮箱地址是什么?"))
""",
)
