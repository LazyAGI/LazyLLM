# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.tools)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.tools)
add_example = functools.partial(utils.add_example, module=lazyllm.tools)

# functions for lazyllm.tools.tools
add_tools_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.tools.tools)
add_tools_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.tools.tools)
add_tools_example = functools.partial(utils.add_example, module=lazyllm.tools.tools)

# ---------------------------------------------------------------------------- #

# rag/document.py

add_english_doc('Document', '''\
Initialize a document module with an optional user interface.

This constructor initializes a document module that can have an optional user interface. If the user interface is enabled, it also provides a UI to manage the document operation interface and offers a web page for user interface interaction.

Args:
    dataset_path (str): The path to the dataset directory. This directory should contain the documents to be managed by the document module.
    embed (Optional[Union[Callable, Dict[str, Callable]]]): The object used to generate document embeddings. If you need to generate multiple embeddings for the text, you need to specify multiple embedding models in a dictionary format. The key identifies the name corresponding to the embedding, and the value is the corresponding embedding model.
    create_ui (bool): [Deprecated] Whether to create a user interface. Use 'manager' parameter instead.
    create_ui (bool): [Deprecated] Whether to create a user interface. Use 'manager' parameter instead.
    manager (bool, optional): A flag indicating whether to create a user interface for the document module. Defaults to False.
    server (Union[bool, int]): Server configuration. True for default server, False for no server, or an integer port number for custom server.
    name (Optional[str]): Name identifier for this document collection. Required for cloud services.
    server (Union[bool, int]): Server configuration. True for default server, False for no server, or an integer port number for custom server.
    name (Optional[str]): Name identifier for this document collection. Required for cloud services.
    launcher (optional): An object or function responsible for launching the server module. If not provided, the default asynchronous launcher from `lazyllm.launchers` is used (`sync=False`).
    doc_fields (optional): Configure the fields that need to be stored and retrieved along with their corresponding types (currently only used by the Milvus backend).
    doc_files (Optional[List[str]]): List of temporary document files (alternative to dataset_path).When used, dataset_path must be None and only map store is supported.
    store_conf (optional): Configure which storage backend and index backend to use.      
    doc_files (Optional[List[str]]): List of temporary document files (alternative to dataset_path).When used, dataset_path must be None and only map store is supported.
    store_conf (optional): Configure which storage backend and index backend to use.      
''')

add_chinese_doc('Document', '''\
初始化一个具有可选用户界面的文档模块。

此构造函数初始化一个可以有或没有用户界面的文档模块。如果启用了用户界面，它还会提供一个ui界面来管理文档操作接口，并提供一个用于用户界面交互的网页。

Args:
    dataset_path (str): 数据集目录的路径。此目录应包含要由文档模块管理的文档。
    embed (Optional[Union[Callable, Dict[str, Callable]]]): 用于生成文档 embedding 的对象。如果需要对文本生成多个 embedding，此处需要通过字典的方式指定多个 embedding 模型，key 标识 embedding 对应的名字, value 为对应的 embedding 模型。
    create_ui (bool):[已弃用] 是否创建用户界面。请改用'manager'参数
    create_ui (bool):[已弃用] 是否创建用户界面。请改用'manager'参数
    manager (bool, optional): 指示是否为文档模块创建用户界面的标志。默认为 False。
    server (Union[bool, int]):服务器配置。True表示默认服务器，False表示已指定端口号作为自定义服务器
    name (Optional[str]):文档集合的名称标识符。云服务模式下必须提供
    launcher (optional): 负责启动服务器模块的对象或函数。如果未提供，则使用 `lazyllm.launchers` 中的默认异步启动器 (`sync=False`)。            
    doc_files (Optional[List[str]]):临时文档文件列表（dataset_path的替代方案）。使用时dataset_path必须为None且仅支持map存储类型
    server (Union[bool, int]):服务器配置。True表示默认服务器，False表示已指定端口号作为自定义服务器
    name (Optional[str]):文档集合的名称标识符。云服务模式下必须提供
    launcher (optional): 负责启动服务器模块的对象或函数。如果未提供，则使用 `lazyllm.launchers` 中的默认异步启动器 (`sync=False`)。            
    doc_files (Optional[List[str]]):临时文档文件列表（dataset_path的替代方案）。使用时dataset_path必须为None且仅支持map存储类型
    store_conf (optional): 配置使用哪种存储后端和索引后端。
''')

add_example('Document', '''\
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)  # or documents = Document(dataset_path='your_doc_path', embed={"key": m}, manager=False)
>>> m1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
>>> document1 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, manager=False)

>>> store_conf = {
>>>     'type': 'chroma',
>>>     'indices': {
>>>         'smart_embedding_index': {
>>>             'backend': 'milvus',
>>>             'kwargs': {
>>>                 'uri': '/tmp/tmp.db',
>>>                 'index_kwargs': {
>>>                     'index_type': 'HNSW',
>>>                     'metric_type': 'COSINE'
>>>                  }
>>>             },
>>>         },
>>>     },
>>> }
>>> doc_fields = {
>>>     'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
>>>     'public_year': DocField(data_type=DataType.INT32),
>>> }
>>> document2 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, store_conf=store_conf, doc_fields=doc_fields)
''')

add_english_doc('Document.create_node_group', '''
Generate a node group produced by the specified rule.

Args:
    name (str): The name of the node group.
    transform (Callable): The transformation rule that converts a node into a node group. The function prototype is `(DocNode, group_name, **kwargs) -> List[DocNode]`. Currently built-in options include [SentenceSplitter][lazyllm.tools.SentenceSplitter], and users can define their own transformation rules.
    trans_node (bool): Determines whether the input and output of transform are `DocNode` or `str`, default is None. Can only be set to true when `transform` is `Callable`.
    num_workers (int): number of new threads used for transform. default: 0
    parent (str): The node that needs further transformation. The series of new nodes obtained after transformation will be child nodes of this parent node. If not specified, the transformation starts from the root node.
    kwargs: Parameters related to the specific implementation.
''')

add_chinese_doc('Document.create_node_group', '''
创建一个由指定规则生成的 node group。

Args:
    name (str): node group 的名称。
    transform (Callable): 将 node 转换成 node group 的转换规则，函数原型是 `(DocNode, group_name, **kwargs) -> List[DocNode]`。目前内置的有 [SentenceSplitter][lazyllm.tools.SentenceSplitter]。用户也可以自定义转换规则。
    trans_node (bool): 决定了transform的输入和输出是 `DocNode` 还是 `str` ，默认为None。只有在 `transform` 为 `Callable` 时才可以设置为true。
    num_workers (int): Transform时所用的新线程数量，默认为0
    parent (str): 需要进一步转换的节点。转换之后得到的一系列新的节点将会作为该父节点的子节点。如果不指定则从根节点开始转换。
    kwargs: 和具体实现相关的参数。
''')

add_example('Document.create_node_group', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
''')

add_english_doc('Document.find_parent', '''
Find the parent node of the specified node.

Args:
    group (str): The name of the node for which to find the parent.
''')

add_chinese_doc('Document.find_parent', '''
查找指定节点的父节点。

Args:
    group (str): 需要查找的节点名称
''')

add_example('Document.find_parent', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
>>> documents.find_parent('children')
''')

add_english_doc('Document.find_children', '''
Find the child nodes of the specified node.

Args:
    group (str): The name of the node for which to find the children.
''')

add_chinese_doc('Document.find_children', '''
查找指定节点的子节点。

Args:
    group (str): 需要查找的名称
''')

add_example('Document.find_children', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
>>> documents.find_children('parent')
''')

add_english_doc('Document.register_global_reader', '''
Used to specify a file reader, which is visible to all Document objects. The registered file reader must be a Callable object. It can be registered using a decorator or by a function call.

Args:
    pattern (str): Matching rules applied by the file reader.
    func (Callable): File reader, must be a Callable object.
''')

add_chinese_doc('Document.register_global_reader', '''
用于指定文件读取器，作用范围对于所有的 Document 对象都可见。注册的文件读取器必须是 Callable 对象。可以使用装饰器的方式进行注册，也可以通过函数调用的方式进行注册。

Args:
    pattern (str): 文件读取器适用的匹配规则
    func (Callable): 文件读取器，必须是Callable的对象
''')

add_example('Document.register_global_reader', '''
>>> from lazyllm.tools.rag import Document, DocNode
>>> @Document.register_global_reader("**/*.yml")
>>> def processYml(file):
...     with open(file, 'r') as f:
...         data = f.read()
...     return [DocNode(text=data)]
...
>>> doc1 = Document(dataset_path="your_files_path", create_ui=False)
>>> doc2 = Document(dataset_path="your_files_path", create_ui=False)
>>> files = ["your_yml_files"]
>>> docs1 = doc1._impl._reader.load_data(input_files=files)
>>> docs2 = doc2._impl._reader.load_data(input_files=files)
>>> print(docs1[0].text == docs2[0].text)
# True
''')

add_english_doc('Document.add_reader', '''
Used to specify the file reader for an instance. The scope of action is visible only to the registered Document object. The registered file reader must be a Callable object. It can only be registered by calling a function. The priority of the file reader registered by the instance is higher than that of the file reader registered by the class, and the priority of the file reader registered by the instance and class is higher than the system default file reader. That is, the order of priority is: instance file reader > class file reader > system default file reader.

Args:
    pattern (str): Matching rules applied by the file reader.
    func (Callable): File reader, must be a Callable object.
''')

add_chinese_doc('Document.add_reader', '''
用于实例指定文件读取器，作用范围仅对注册的 Document 对象可见。注册的文件读取器必须是 Callable 对象。只能通过函数调用的方式进行注册。并且通过实例注册的文件读取器的优先级高于通过类注册的文件读取器，并且实例和类注册的文件读取器的优先级高于系统默认的文件读取器。即优先级的顺序是：实例文件读取器 > 类文件读取器 > 系统默认文件读取器。

Args:
    pattern (str): 文件读取器适用的匹配规则
    func (Callable): 文件读取器，必须是Callable的对象
''')

add_example('Document.add_reader', '''
>>> from lazyllm.tools.rag import Document, DocNode
>>> from lazyllm.tools.rag.readers import ReaderBase
>>> class YmlReader(ReaderBase):
...     def _load_data(self, file, fs=None):
...         try:
...             import yaml
...         except ImportError:
...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
...         with open(file, 'r') as f:
...             data = yaml.safe_load(f)
...         print("Call the class YmlReader.")
...         return [DocNode(text=data)]
...
>>> def processYml(file):
...     with open(file, 'r') as f:
...         data = f.read()
...     print("Call the function processYml.")
...     return [DocNode(text=data)]
...
>>> doc1 = Document(dataset_path="your_files_path", create_ui=False)
>>> doc2 = Document(dataset_path="your_files_path", create_ui=False)
>>> doc1.add_reader("**/*.yml", YmlReader)
>>> print(doc1._impl._local_file_reader)
{'**/*.yml': <class '__main__.YmlReader'>}
>>> print(doc2._impl._local_file_reader)
{}
>>> files = ["your_yml_files"]
>>> Document.register_global_reader("**/*.yml", processYml)
>>> doc1._impl._reader.load_data(input_files=files)
Call the class YmlReader.
>>> doc2._impl._reader.load_data(input_files=files)
Call the function processYml.
''')

add_english_doc('rag.readers.ReaderBase', '''
The base class of file readers, which inherits from the ModuleBase base class and has Callable capabilities. Subclasses that inherit from this class only need to implement the _load_data function, and its return parameter type is List[DocNode]. Generally, the input parameters of the _load_data function are file (Path) and fs (AbstractFileSystem).

Args:
    args (Any): Pass the corresponding position parameters as needed.
    return_trace (bool): Set whether to record trace logs.
    kwargs (Dict): Pass the corresponding keyword arguments as needed.
''')

add_chinese_doc('rag.readers.ReaderBase', '''
文件读取器的基类，它继承自 ModuleBase 基类，具有 Callable 的能力，继承自该类的子类只需要实现 _load_data 函数即可，它的返回参数类型为 List[DocNode]. 一般 _load_data 函数的入参为 file (Path), fs(AbstractFileSystem) 三个参数。

Args:
    args (Any): 根据需要传输相应的位置参数
    return_trace (bool): 设置是否记录trace日志
    kwargs (Dict): 根据需要传输相应的关键字参数
''')

add_example('rag.readers.ReaderBase', '''
>>> from lazyllm.tools.rag.readers import ReaderBase
>>> from lazyllm.tools.rag import DocNode, Document
>>> from typing import Dict, Optional, List
>>> from pathlib import Path
>>> from fsspec import AbstractFileSystem
>>> @Document.register_global_reader("**/*.yml")
>>> class YmlReader(ReaderBase):
...     def _load_data(self, file: Path, fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
...         try:
...             import yaml
...         except ImportError:
...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
...         with open(file, 'r') as f:
...             data = yaml.safe_load(f)
...         print("Call the class YmlReader.")
...         return [DocNode(text=data)]
...
>>> files = ["your_yml_files"]
>>> doc = Document(dataset_path="your_files_path", create_ui=False)
>>> reader = doc._impl._reader.load_data(input_files=files)
# Call the class YmlReader.
''')

add_chinese_doc('rag.component.bm25.BM25', '''\
基于 BM25 算法实现的检索器，用于从节点集合中根据查询词检索最相关的文本节点。

Args:
    nodes (List[DocNode]): 需要建立索引的文本节点列表。
    language (str): 所使用的语言，支持 ``en``（英文）或 ``zh``（中文）。默认为 ``en``。
    topk (int): 每次检索返回的最大节点数量，默认值为2。
''')

add_english_doc('rag.component.bm25.BM25', '''\
A retriever based on the BM25 algorithm that retrieves the most relevant text nodes from a given list of nodes.

Args:
    nodes (List[DocNode]): A list of text nodes to index.
    language (str): The language to use, supports ``en`` (English) and ``zh`` (Chinese). Defaults to ``en``.
    topk (int): The maximum number of nodes to return in each retrieval. Defaults to 2.
''')

add_chinese_doc('rag.doc_to_db.DocInfoSchemaItem', '''\
文档信息结构中单个字段的定义。

Args:
    key (str): 字段名
    desc (str): 字段含义描述
    type (str): 字段的数据类型
''')

add_english_doc('rag.doc_to_db.DocInfoSchemaItem', '''\
Definition of a single field in the document information schema.

Args:
    key (str): The name of the field.
    desc (str): The description of the field's meaning.
    type (str): The data type of the field.
''')

add_chinese_doc('rag.doc_to_db.DocGenreAnalyser', '''\
用于分析文档所属的类别，例如合同、简历、发票等。通过读取文档内容，并结合大模型判断其类型。

Args:
    maximum_doc_num (int): 最多分析的文档数量，默认是 3。
''')

add_english_doc('rag.doc_to_db.DocGenreAnalyser', '''\
Used to analyze the genre/type of documents, such as contracts, resumes, invoices, etc. It reads the document content and uses a language model to classify its type.

Args:
    maximum_doc_num (int): Maximum number of documents to analyze, default is 3.
''')

add_example('rag.doc_to_db.DocGenreAnalyser', '''\
>>> import lazyllm
>>> from lazyllm.components.doc_info_extractor import DocGenreAnalyser
>>> from lazyllm import OnlineChatModule
>>> m = OnlineChatModule(source="openai")
>>> analyser = DocGenreAnalyser()
>>> genre = analyser.analyse_doc_genre(m, "path/to/document.txt")
>>> print(genre)
contract
''')

add_chinese_doc('rag.doc_to_db.DocInfoSchemaAnalyser', '''\
用于从文档中抽取出关键信息字段的结构，如字段名、描述、字段类型。可用于构建信息提取模板。

Args:
    maximum_doc_num (int): 用于生成schema的最大文档数量，默认是 3。
''')

add_english_doc('rag.doc_to_db.DocInfoSchemaAnalyser', '''\
Used to extract key-value schema from documents, such as field names, descriptions, and data types. Useful for building structured information extraction templates.

Args:
    maximum_doc_num (int): Maximum number of documents to be used for generating schema, default is 3.
''')

add_example('rag.doc_to_db.DocInfoSchemaAnalyser', '''\
>>> from lazyllm.components.doc_info_extractor import DocInfoSchemaAnalyser
>>> from lazyllm import OnlineChatModule
>>> analyser = DocInfoSchemaAnalyser()
>>> m = OnlineChatModule(source="openai")
>>> schema = analyser.analyse_info_schema(m, "contract", ["doc1.txt", "doc2.txt"])
>>> print(schema)
[{'key': 'party_a', 'desc': 'The first party', 'type': 'str'}, ...]
''')

add_chinese_doc('rag.doc_to_db.DocInfoExtractor', '''\
根据给定的字段结构（schema）从文档中抽取具体的关键信息值，返回格式为 key-value 字典。

Args:
    无
''')

add_english_doc('rag.doc_to_db.DocInfoExtractor', '''\
Extracts specific values for key fields from a document according to a provided schema. Returns a dictionary of key-value pairs.

Args:
    None
''')

add_example('rag.doc_to_db.DocInfoExtractor', '''\
>>> from lazyllm.components.doc_info_extractor import DocInfoExtractor
>>> from lazyllm import OnlineChatModule
>>> extractor = DocInfoExtractor()
>>> m = OnlineChatModule(source="openai")
>>> schema = [{"key": "party_a", "desc": "Party A name", "type": "str"}]
>>> info = extractor.extract_doc_info(m, "contract.txt", schema)
>>> print(info)
{'party_a': 'ABC Corp'}
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor', '''\
用于将文档信息抽取并导出到数据库中。

该类通过分析文档主题、抽取字段结构、从文档中提取关键信息，并将其保存至数据库表中。

Args:
    sql_manager (SqlManager): 数据库管理模块。
    doc_table_name (str): 存储文档字段的数据库表名，默认为`lazyllm_doc_elements`。

Note:
    - 如果表已存在，会自动检测并避免重复创建。
    - 如果你希望重置字段结构，使用 `reset_doc_info_schema` 方法。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor', '''\
Used to extract information from documents and export it to a database.

This class analyzes document topics, extracts schema structure, pulls out key information, and saves it into a database table.

Args:
    sql_manager (SqlManager): The SQL management module.
    doc_table_name (str): The table name to store document fields. Default is ``lazyllm_doc_elements``.

Note:
    - If the table already exists, it checks and avoids redundant creation.
    - Use `reset_doc_info_schema` to reset the schema if necessary.
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.extract_info_from_docs', '''\
从文档中提取结构化数据库信息。

该函数使用嵌入和检索技术，在提供的文档中获取数据库相关的文本片段，用于后续模式生成。

Args:
    docs (list[DocNode]): 输入文档列表。
    num_nodes (int): 要提取的片段数量，默认为10。

Returns:
    list[DocNode]: 提取出的相关文档片段。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor.extract_info_from_docs', '''\
Extract structured database-related information from documents.

This function uses embedding and retrieval techniques to identify relevant text fragments in the provided documents for schema generation.

Args:
    docs (list[DocNode]): List of input documents.
    num_nodes (int): Number of text fragments to retrieve. Default is 10.

Returns:
    list[DocNode]: The relevant extracted document nodes.
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.analyze_info_schema_by_llm', '''\
使用大语言模型从文档节点中推断数据库信息结构。

Args:
    nodes (list[DocNode]): 文档节点列表。

Returns:
    dict: 结构化信息模式，包含表名、字段、关系等信息。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor.analyze_info_schema_by_llm', '''\
Infer structured database information using a large language model from document nodes.

Args:
    nodes (list[DocNode]): List of document nodes.

Returns:
    dict: The inferred database schema, including table names, fields, and relationships.
''')


add_chinese_doc('rag.doc_to_db.extract_db_schema_from_files', '''\
给定文档路径和LLM模型，提取文档结构信息。

Args:
    file_paths (List[str]): 要分析的文档路径。
    llm (Union[OnlineChatModule, TrainableModule]): 支持聊天的模型模块。

Returns:
    DocInfoSchema: 提取出的字段结构描述。
''')

add_english_doc('rag.doc_to_db.extract_db_schema_from_files', '''\
Extract the schema information from documents using a given LLM.

Args:
    file_paths (List[str]): Paths of the documents to analyze.
    llm (Union[OnlineChatModule, TrainableModule]): A chat-supported LLM module.

Returns:
    DocInfoSchema: The extracted field structure schema.
''')

add_example('rag.doc_to_db.extract_db_schema_from_files', '''\
>>> import lazyllm
>>> from lazyllm.components.document_to_db import extract_db_schema_from_files
>>> llm = lazyllm.OnlineChatModule()
>>> file_paths = ["doc1.pdf", "doc2.pdf"]
>>> schema = extract_db_schema_from_files(file_paths, llm)
>>> print(schema)
''')

add_chinese_doc('rag.readers.DocxReader', """\
docx格式文件解析器，从 `.docx` 文件中读取文本内容并封装为文档节点（DocNode）列表。

Args:
    file (Path): `.docx` 文件路径。
    fs (Optional[AbstractFileSystem]): 可选的文件系统对象，支持自定义读取方式。

Returns:
    List[DocNode]: 包含文档中所有文本内容的节点列表。
""")

add_english_doc('rag.readers.DocxReader', """\
A docx format file parser, reading text content from a `.docx` file and return a list of `DocNode` objects.

Args:
    file (Path): Path to the `.docx` file.
    fs (Optional[AbstractFileSystem]): Optional file system object for custom reading.

Returns:
    List[DocNode]: A list containing the extracted text content as `DocNode` instances.
""")

add_chinese_doc('rag.readers.EpubReader', """\
用于读取 `.epub` 格式电子书的文件读取器。

继承自 `LazyLLMReaderBase`，只需实现 `_load_data` 方法，即可通过 `Document` 组件自动加载 `.epub` 文件中的内容。

注意：当前版本不支持通过 fsspec 文件系统（如远程路径）加载 epub 文件，若提供 `fs` 参数，将回退到本地文件读取。

Returns:
    List[DocNode]: 所有章节内容合并后的文本节点列表。
""")

add_english_doc('rag.readers.EpubReader', """\
A file reader for `.epub` format eBooks.

Inherits from `LazyLLMReaderBase`, and only needs to implement `_load_data`. The `Document` module can automatically use this class to load `.epub` files.

Note: Reading from fsspec file systems (e.g., remote paths) is not supported in this version. If `fs` is specified, it will fall back to reading from the local file system.

Returns:
    List[DocNode]: A single node containing all merged chapter content from the EPUB file.
""")

add_chinese_doc('rag.readers.HWPReader', '''\
HWP文件解析器，支持从本地文件系统读取 HWP 文件。它会从文档中提取正文部分的文本内容，返回 DocNode 列表。

HWP 是一种专有的二进制格式，主要在韩国使用。由于格式封闭，因此只能解析部分内容（如文本段落），但对常规文本提取已经足够使用。

Args:
    return_trace (bool): 是否启用 trace 日志记录，默认为 ``True``。
''')

add_english_doc('rag.readers.HWPReader', '''
A HWP format file parser. It supports loading from the local filesystem. It extracts body text from the `.hwp` file and returns it as a list of DocNode objects.

HWP is a proprietary binary document format used primarily in Korea. This reader focuses on extracting the plain text from the body sections of the document.

Args:
    return_trace (bool): Whether to enable trace logging. Defaults to ``True``.
''')

add_chinese_doc('rag.readers.ImageReader', '''\
用于从图片文件中读取内容的模块。支持保留图片、解析图片中的文本（基于OCR或预训练视觉模型），并返回文本和图片路径的节点列表。

Args:
    parser_config (Optional[Dict]): 解析器配置，包含模型和处理器，默认为 None。当设置 parse_text=True 且 parser_config=None 时，会自动根据 text_type 加载相应模型。
    keep_image (bool): 是否保留图片的 base64 编码，默认为 False。
    parse_text (bool): 是否解析图片中的文本，默认为 False。
    text_type (str): 解析文本的类型，支持 ``text``（默认）和 ``plain_text``。当为 ``plain_text`` 时，使用 pytesseract 进行OCR；否则使用预训练视觉编码解码模型。
    pytesseract_model_kwargs (Optional[Dict]): 传递给 pytesseract OCR 的可选参数，默认为空字典。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.ImageReader', '''\
Module for reading content from image files. Supports keeping the image as base64, parsing text from images using OCR or pretrained vision models, and returns a list of nodes with text and image path.

Args:
    parser_config (Optional[Dict]): Parser configuration containing the model and processor. Defaults to None. When parse_text=True and parser_config is None, relevant models will be auto-loaded based on text_type.
    keep_image (bool): Whether to keep the image as base64 string. Default is False.
    parse_text (bool): Whether to parse text from the image. Default is False.
    text_type (str): Type of text parsing. Supports ``text`` (default) and ``plain_text``. If ``plain_text``, pytesseract OCR is used; otherwise a pretrained vision encoder-decoder model is used.
    pytesseract_model_kwargs (Optional[Dict]): Optional arguments passed to pytesseract OCR. Defaults to empty dict.
    return_trace (bool): Whether to record the processing trace. Default is True.
''')

add_chinese_doc('rag.readers.IPYNBReader', '''\
用于读取和解析 Jupyter Notebook (.ipynb) 文件的模块。将 notebook 转换成脚本文本后，按代码单元划分为多个文档节点，或合并为单一文本节点。

Args:
    parser_config (Optional[Dict]): 预留的解析器配置参数，当前未使用，默认为 None。
    concatenate (bool): 是否将所有代码单元合并成一个整体文本节点，默认为 False，即分割为多个节点。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.IPYNBReader', '''\
Module for reading and parsing Jupyter Notebook (.ipynb) files. Converts the notebook to script text, then splits it by code cells into multiple document nodes or concatenates into a single text node.

Args:
    parser_config (Optional[Dict]): Reserved parser configuration parameter, currently unused. Defaults to None.
    concatenate (bool): Whether to concatenate all code cells into one text node. Defaults to False (split into multiple nodes).
    return_trace (bool): Whether to record processing trace. Default is True.
''')

add_chinese_doc('rag.readers.MagicPDFReader', '''\
用于通过 MagicPDF 服务解析 PDF 文件内容的模块。支持上传文件或通过 URL 方式调用解析接口，解析结果经过回调函数处理成文档节点列表。

Args:
    magic_url (str): MagicPDF 服务的接口 URL。
    callback (Optional[Callable[[List[dict], Path, dict], List[DocNode]]]): 解析结果回调函数，接收解析元素列表、文件路径及额外信息，返回文档节点列表。默认将所有文本合并为一个节点。
    upload_mode (bool): 是否采用文件上传模式调用接口，默认为 False，即通过 JSON 请求文件路径。
''')

add_english_doc('rag.readers.MagicPDFReader', '''\
Module to parse PDF content via the MagicPDF service. Supports file upload or URL-based parsing, with a callback to process the parsed elements into document nodes.

Args:
    magic_url (str): The MagicPDF service API URL.
    callback (Optional[Callable[[List[dict], Path, dict], List[DocNode]]]): A callback function that takes parsed element list, file path, and extra info, returns a list of DocNode. Defaults to merging all text into a single node.
    upload_mode (bool): Whether to use file upload mode for the API call. Default is False, meaning JSON request with file path.
''')

add_chinese_doc('rag.readers.MarkdownReader', '''\
用于读取和解析 Markdown 文件的模块。支持去除超链接和图片，按标题和内容将 Markdown 划分成若干文本段落节点。

Args:
    remove_hyperlinks (bool): 是否移除超链接，默认 True。
    remove_images (bool): 是否移除图片标记，默认 True。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.MarkdownReader', '''\
Module for reading and parsing Markdown files. Supports removing hyperlinks and images, and splits Markdown into text segments by headers, returning document nodes.

Args:
    remove_hyperlinks (bool): Whether to remove hyperlinks, default is True.
    remove_images (bool): Whether to remove image tags, default is True.
    return_trace (bool): Whether to record processing trace, default is True.
''')

add_chinese_doc('rag.readers.MarkdownReader.remove_images', '''\
移除内容中形如 ![[...]] 的自定义图片标签。

Args:
    content (str): 输入的 markdown 内容。

Returns:
    str: 移除图片标签后的内容。
''')

add_english_doc('rag.readers.MarkdownReader.remove_images', '''\
Remove custom image tags of the form ![[...]] from the content.

Args:
    content (str): Input markdown content.

Returns:
    str: Content with image tags removed.
''')

add_chinese_doc('rag.readers.MarkdownReader.remove_hyperlinks', '''\
移除 Markdown 超链接，将 [文本](链接) 转换为纯文本。

Args:
    content (str): 输入的 markdown 内容。

Returns:
    str: 移除超链接后的内容，仅保留链接文本。
''')

add_english_doc('rag.readers.MarkdownReader.remove_hyperlinks', '''\
Remove markdown hyperlinks, converting [text](url) to just text.

Args:
    content (str): Input markdown content.

Returns:
    str: Content with hyperlinks removed, only link text retained.
''')

add_chinese_doc('rag.readers.MboxReader', '''\
用于解析 Mbox 邮件存档文件的模块。读取邮件内容并格式化为文本，支持限制最大邮件数和自定义消息格式。

Args:
    max_count (int): 最大读取的邮件数量，默认 0 表示读取全部邮件。
    message_format (str): 邮件文本格式模板，支持使用 ``{_date}``、``{_from}``、``{_to}``、``{_subject}`` 和 ``{_content}`` 占位符。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.MboxReader', '''\
Module to parse Mbox email archive files. Reads email messages and formats them into text. Supports limiting the maximum number of messages and custom message formatting.

Args:
    max_count (int): Maximum number of emails to read. Default 0 means read all.
    message_format (str): Template string for formatting each message, supports placeholders ``{_date}``, ``{_from}``, ``{_to}``, ``{_subject}``, and ``{_content}``.
    return_trace (bool): Whether to record processing trace. Default is True.
''')


add_english_doc('rag.store.ChromadbStore', '''
Inherits from the abstract base class StoreBase. This class is mainly used to store and manage document nodes (DocNode), supporting operations such as node addition, deletion, modification, query, index management, and persistent storage.
Args:
    group_embed_keys (Dict[str, Set[str]]): Specifies the embedding fields associated with each document group.
    embed (Dict[str, Callable]): A dictionary of embedding generation functions, supporting multiple embedding sources.
    embed_dims (Dict[str, int]): The embedding dimensions corresponding to each embedding type.
    dir (str): Path to the chromadb persistent storage directory.
    kwargs (Dict): Additional optional parameters passed to the parent class or internal components.
''')


add_chinese_doc('rag.store.ChromadbStore', '''
继承自 StoreBase 抽象基类。它主要用于存储和管理文档节点(DocNode)，支持节点增删改查、索引管理和持久化存储。
Args:
     group_embed_keys (Dict[str, Set[str]]): 指定每个文档分组所对应的嵌入字段。
    embed (Dict[str, Callable]): 嵌入生成函数或其映射，支持多嵌入源。
    embed_dims (Dict[str, int]): 每种嵌入类型对应的维度。
    dir (str): chromadb 数据库存储路径。
    kwargs (Dict): 其他可选参数，传递给父类或内部组件。
''')

add_example('rag.store.ChromadbStore', '''
>>> from lazyllm.tools.rag.chroma_store import ChromadbStore
>>> from typing import Dict, List
>>> import numpy as np
>>> store = ChromadbStore(
...     group_embed_keys={"articles": {"title_embed", "content_embed"}},
...     embed={
...         "title_embed": lambda x: np.random.rand(128).tolist(),
...         "content_embed": lambda x: np.random.rand(256).tolist()
...     },
...     embed_dims={"title_embed": 128, "content_embed": 256},
...     dir="./chroma_data"
... )
>>> store.update_nodes([node1, node2])
>>> results = store.query(query_text="文档内容", group_name="articles", top_k=2)
>>> for node in results:
...     print(f"找到文档: {node._content[:20]}...")
>>> store.remove_nodes(doc_ids=["doc1"])
''')

add_english_doc('rag.store.ChromadbStore.update_nodes', '''
Update a group of DocNode objects.
Args:
    nodes (DocNode): The list of DocNode objects to be updated.
''')


add_chinese_doc('rag.store.ChromadbStore.update_nodes', '''
更新一组 DocNode 节点。
Args:
    nodes(DocNode): 需要更新的 DocNode 列表。
''')


add_english_doc('rag.store.ChromadbStore.remove_nodes', '''
Delete nodes based on specified conditions.
Args:
    doc_ids (str): Delete by document ID.
    group_name (str): Specify the group name for deletion.
    uids (str): Delete by unique node ID.
''')


add_chinese_doc('rag.store.ChromadbStore.remove_nodes', '''
删除指定条件的节点。
Args:
    doc_ids(str): 按文档 ID 删除。
    group_name(str): 限定删除的组名。
    uids(str): 按节点唯一 ID 删除。
''')


add_english_doc('rag.store.ChromadbStore.update_doc_meta', '''
Update the metadata of a document.
Args:
    doc_id (str): The ID of the document to be updated.
    metadata (dict): The new metadata (key-value pairs).
''')


add_chinese_doc('rag.store.ChromadbStore.update_doc_meta', '''
更新文档的元数据。。
Args:
    doc_id(str):需要更新的文档 ID。
    metadata(dict):新的元数据（键值对）。
''')


add_english_doc('rag.store.ChromadbStore.get_nodes', '''
Query nodes based on specified conditions.
Args:
    group_name (str): The name of the group to which the nodes belong.
    uids (List[str]): A list of unique node IDs.
    doc_ids (Set[str]): A set of document IDs.
    **kwargs: Additional optional parameters.
''')


add_chinese_doc('rag.store.ChromadbStore.get_nodes', '''
根据条件查询节点。
Args:
    group_name(str]):节点所属的组名。
    uids(List[str]):节点唯一 ID 列表。
    doc_ids	(Set[str])：文档 ID 集合。
    **kwargs:其他扩展参数。
''')


add_english_doc('rag.store.ChromadbStore.activate_group', '''
Activate the specified group.
Args:
    group_names([str, List[str]]): Activate by group name.
''')


add_chinese_doc('rag.store.ChromadbStore.activate_group', '''
激活指定的组。
Args:
    group_names([str, List[str]])：按组名激活。
''')

add_english_doc('rag.store.ChromadbStore.activated_groups', '''
Activate groups. Return the list of currently activated group names.
''')


add_chinese_doc('rag.store.ChromadbStore.activated_groups', '''
激活组，返回当前激活的组名列表。
''')
add_english_doc('rag.store.ChromadbStore.query', '''
Execute a query using the default index.
Args:
    args: Query parameters.
    kwargs: Additional optional parameters.
''')


add_chinese_doc('rag.store.ChromadbStore.query', '''
通过默认索引执行查询。
Args:
    args：查询参数。
    kwargs：其他扩展参数。
''')

add_english_doc('rag.store.ChromadbStore.is_group_active', '''
Check whether the specified group is active.
Args:
    name (str): The name of the group.
''')

add_chinese_doc('rag.store.ChromadbStore.is_group_active', '''
检查指定组是否激活。
Args:
    name(str)：组名。
''')


add_english_doc('rag.store.ChromadbStore.all_groups', '''
Return the list of all group names.
''')


add_chinese_doc('rag.store.ChromadbStore.all_groups', '''
返回所有组名列表。
''')

add_english_doc('rag.store.ChromadbStore.register_index', '''
Register a custom index.
Args:
    type (str): The name of the index type.
    index (IndexBase): An object implementing the IndexBase interface.
''')


add_chinese_doc('rag.store.ChromadbStore.register_index', '''
注册自定义索引。
Args:
    type(str):索引类型名称。
    index(IndexBase):实现 IndexBase 的对象。
''')


add_english_doc('rag.store.ChromadbStore.get_index', '''
Get the index of the specified type.
Args:
    type (str): The type of the index.
''')


add_chinese_doc('rag.store.ChromadbStore.get_index', '''
获取指定类型的索引。
Args:
    type(str):索引类型
''')


add_english_doc('rag.store.ChromadbStore.clear_cache', '''
Clear the ChromaDB collections and memory cache for specified groups or all groups.
Args:
    group_names (List[str]): List of group names. If None, clear all groups.
''')


add_chinese_doc('rag.store.ChromadbStore.clear_cache', '''
清除指定组或所有组的 ChromaDB 集合和内存缓存。
Args:
    group_names(List[str])：组名列表，为 None 时清除所有组。
''')





add_english_doc('rag.store.MilvusStore', '''
Inherits from the StoreBase abstract base class. Implements a vector database based on Milvus. Its functionality is similar to ChromadbStore, used for storing, managing, indexing, and querying embedded document nodes (DocNode).
Args:
    group_embed_keys (Dict[str, Set[str]]): Specifies the embedding fields for each group.
    embed (Dict[str, Callable]): Embedding functions for each field.
    embed_dims (Dict[str, int]): Vector dimensions for each embedding field.
    embed_datatypes (Dict[str, DataType]): Vector types for each embedding field (must comply with Milvus types).
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Description of global metadata fields, used to configure other non-vector fields in Milvus.
    url (str): Milvus connection address, supporting local or remote connections.
    index_kwargs (Union[Dict, List]): Optional index parameters for creating Milvus vector indexes, such as IVF, HNSW parameters.
    db_name (str): Optional, defaults to "lazyllm". Represents the database name in Milvus.
''')


add_chinese_doc('rag.store.MilvusStore', '''
继承自 StoreBase 抽象基类。基于 Milvus 向量数据库实现。其功能和 ChromadbStore 类似, 用于存储、管理、索引和查询嵌入向量化后的文档节点(DocNode)。
Args:
    group_embed_keys (Dict[str, Set[str]]): 指定每个group所对应的嵌入字段。
    embed (Dict[str, Callable]): 每种字段对应的 embedding 函数.
    embed_dims (Dict[str, int]): 每个嵌入字段的向量维度。
    embed_datatypes(Dict[str, DataType]): 每个嵌入字段的向量类型（需符合 Milvus 类型）。
    global_metadata_descDict([str, GlobalMetadataDesc])：全局元数据字段的说明，用于配置 Milvus 中的其他非向量字段。
    url(str):Milvus 的连接地址，支持本地或远程。
    index_kwargs:([Union[Dict, List]]):可选的索引参数，用于创建 Milvus 的向量索引，例如 IVF、HNSW 参数。
    db_name(str):可选，默认 "lazyllm"。表示 Milvus 中的数据库名。
''')

add_example('rag.store.MilvusStore', '''
>>> from lazyllm.tools.rag.milvus_store import MilvusStore
>>> from typing import Dict, List
>>> import numpy as np
>>> store = MilvusStore(
...     group_embed_keys={
...         "articles": {"text"},
...         "faqs": {"question"}
...     },
...     embed={
...         "text": lambda x: np.random.rand(128).tolist(),
...         "question": lambda x: np.random.rand(128).tolist()
...     },
...     embed_dims={"text": 128, "question": 128},
...     embed_datatypes={"text": DataType.FLOAT_VECTOR, "question": DataType.FLOAT_VECTOR},
...     global_metadata_desc=None,
...     uri="http://localhost:19530",
...     index_kwargs={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
...     db_name="test_db"
... )
>>> store.update_nodes([node1, node2])
>>> results = store.query(query_text="文档内容", group_name="articles", top_k=2)
>>> for node in results:
...     print(f"找到文档: {node._content[:20]}...")
>>> store.remove_nodes(doc_ids=["doc1"])
''')

add_english_doc('rag.store.MilvusStore.update_nodes', '''
Update or insert nodes into Milvus collections and memory store.
Args:
    nodes (List[DocNode]): List of document nodes to update.
''')

add_chinese_doc('rag.store.MilvusStore.update_nodes', '''
更新或插入节点到 Milvus 集合和内存存储中。
Args:
    nodes (List[DocNode]): 需要更新的文档节点列表。
''')

add_english_doc('rag.store.MilvusStore.update_doc_meta', '''
Update metadata for a document and sync to all related nodes.
Args:
    doc_id (str): Target document ID.
    metadata (dict): New metadata key-value pairs.
''')

add_chinese_doc('rag.store.MilvusStore.update_doc_meta', '''
更新文档元数据并同步到所有关联节点。
Args:
    doc_id (str): 目标文档ID。
    metadata (dict): 新的元数据键值对。
''')

add_english_doc('rag.store.MilvusStore.remove_nodes', '''
Remove nodes by document IDs, group name, or node UIDs.
Args:
    doc_ids (Optional[List[str]]): Document IDs filter.
    group_name (Optional[str]): Group name filter.
    uids (Optional[List[str]]): Node UIDs filter.
''')

add_chinese_doc('rag.store.MilvusStore.remove_nodes', '''
通过文档ID、组名或节点UID删除节点。
Args:
    doc_ids (Optional[List[str]]): 文档ID过滤条件。
    group_name (Optional[str]): 组名过滤条件。
    uids (Optional[List[str]]): 节点UID过滤条件。
''')
add_english_doc('rag.store.MilvusStore.get_nodes', '''
Query nodes with flexible filtering options.
Args:
    group_name (Optional[str]): Group name filter.
    uids (Optional[List[str]]): Node UIDs filter.
    doc_ids (Optional[Set[str]]): Document IDs filter.
    **kwargs: Additional query parameters.
Returns:
    List[DocNode]: Matched document nodes.
''')

add_chinese_doc('rag.store.MilvusStore.get_nodes', '''
通过多条件查询节点。
Args:
    group_name (Optional[str]): 组名过滤条件。
    uids (Optional[List[str]]): 节点UID过滤条件。
    doc_ids (Optional[Set[str]]): 文档ID过滤条件。
    **kwargs: 其他查询参数。
Returns:
    List[DocNode]: 匹配的文档节点列表。
''')

add_english_doc('rag.store.MilvusStore.query', '''
Semantic search with vector similarity.
Args:
    query (str): Query text.
    group_name (str): Target group name.
    similarity_cut_off (Optional[Union[float, Dict[str, float]]]): Similarity threshold.
    topk (int): Number of results to return.
    embed_keys (List[str]): Embedding keys for search.
    filters (Optional[Dict]): Metadata filters.
Returns:
    List[DocNode]: Nodes with similarity scores.
''')

add_chinese_doc('rag.store.MilvusStore.query', '''
基于向量相似度的语义搜索。
Args:
    query (str): 查询文本。
    group_name (str): 目标组名。
    similarity_cut_off (Optional[Union[float, Dict[str, float]]): 相似度阈值。
    topk (int): 返回结果数量。
    embed_keys (List[str]): 用于搜索的嵌入键。
    filters (Optional[Dict]): 元数据过滤条件。
Returns:
    List[DocNode]: 带相似度分数的节点列表。
''')

add_english_doc('rag.store.MilvusStore.activate_group', '''
Activate one or multiple groups for operations.
Args:
    group_names (Union[str, List[str]]): Group name(s) to activate.
''')

add_chinese_doc('rag.store.MilvusStore.activate_group', '''
激活一个或多个组用于后续操作。
Args:
    group_names (Union[str, List[str]]): 要激活的组名（单个或列表）。
''')

add_english_doc('rag.store.MilvusStore.get_index', '''
Get index instance by type.
Args:
    type (Optional[str]): Index type name, defaults to "default".
''')

add_chinese_doc('rag.store.MilvusStore.get_index', '''
获取指定类型的索引实例。
Args:
    type (Optional[str]): 索引类型名称，默认为"default"。
''')

add_english_doc('rag.store.MilvusStore.register_index', '''
Register custom index type.
Args:
    type (str): Index type name.
    index (IndexBase): Custom index instance.
''')

add_chinese_doc('rag.store.MilvusStore.register_index', '''
注册自定义索引类型。
Args:
    type (str): 索引类型名称。
    index (IndexBase): 自定义索引实例。
''')

add_english_doc('rag.store.MilvusStore.activated_groups', '''
Get names of all activated groups.
Returns:
    List[str]: Active group names.
''')

add_chinese_doc('rag.store.MilvusStore.activated_groups', '''
获取所有已激活的组名。
Returns:
    List[str]: 活跃组名列表。
''')

add_english_doc('rag.store.MilvusStore.is_group_active', '''
Check if a group is activated.
Args:
    name (str): Group name to check.
''')

add_chinese_doc('rag.store.MilvusStore.is_group_active', '''
检查指定组是否激活。
Args:
    name (str): 要检查的组名。
''')

add_chinese_doc('rag.component.bm25.BM25', '''\
基于 BM25 算法实现的检索器，用于从节点集合中根据查询词检索最相关的文本节点。

Args:
    nodes (List[DocNode]): 需要建立索引的文本节点列表。
    language (str): 所使用的语言，支持 ``en``（英文）或 ``zh``（中文）。默认为 ``en``。
    topk (int): 每次检索返回的最大节点数量，默认值为2。
''')

add_english_doc('rag.component.BM25', '''\
A retriever based on the BM25 algorithm that retrieves the most relevant text nodes from a given list of nodes.

Args:
    nodes (List[DocNode]): A list of text nodes to index.
    language (str): The language to use, supports ``en`` (English) and ``zh`` (Chinese). Defaults to ``en``.
    topk (int): The maximum number of nodes to return in each retrieval. Defaults to 2.
''')

add_chinese_doc('rag.doc_to_db.DocInfoSchemaItem', '''\
文档信息结构中单个字段的定义。

Args:
    key (str): 字段名
    desc (str): 字段含义描述
    type (str): 字段的数据类型
''')

add_english_doc('rag.doc_to_db.DocInfoSchemaItem', '''\
Definition of a single field in the document information schema.

Args:
    key (str): The name of the field.
    desc (str): The description of the field's meaning.
    type (str): The data type of the field.
''')

add_chinese_doc('rag.doc_to_db.DocGenreAnalyser', '''\
用于分析文档所属的类别，例如合同、简历、发票等。通过读取文档内容，并结合大模型判断其类型。

Args:
    maximum_doc_num (int): 最多分析的文档数量，默认是 3。
''')

add_english_doc('rag.doc_to_db.DocGenreAnalyser', '''\
Used to analyze the genre/type of documents, such as contracts, resumes, invoices, etc. It reads the document content and uses a language model to classify its type.

Args:
    maximum_doc_num (int): Maximum number of documents to analyze, default is 3.
''')

add_example('rag.doc_to_db.DocGenreAnalyser', '''\
>>> import lazyllm
>>> from lazyllm.components.doc_info_extractor import DocGenreAnalyser
>>> from lazyllm import OnlineChatModule
>>> m = OnlineChatModule(source="openai")
>>> analyser = DocGenreAnalyser()
>>> genre = analyser.analyse_doc_genre(m, "path/to/document.txt")
>>> print(genre)
contract
''')

add_chinese_doc('rag.doc_to_db.DocInfoSchemaAnalyser', '''\
用于从文档中抽取出关键信息字段的结构，如字段名、描述、字段类型。可用于构建信息提取模板。

Args:
    maximum_doc_num (int): 用于生成schema的最大文档数量，默认是 3。
''')

add_english_doc('rag.doc_to_db.DocInfoSchemaAnalyser', '''\
Used to extract key-value schema from documents, such as field names, descriptions, and data types. Useful for building structured information extraction templates.

Args:
    maximum_doc_num (int): Maximum number of documents to be used for generating schema, default is 3.
''')

add_example('rag.doc_to_db.DocInfoSchemaAnalyser', '''\
>>> from lazyllm.components.doc_info_extractor import DocInfoSchemaAnalyser
>>> from lazyllm import OnlineChatModule
>>> analyser = DocInfoSchemaAnalyser()
>>> m = OnlineChatModule(source="openai")
>>> schema = analyser.analyse_info_schema(m, "contract", ["doc1.txt", "doc2.txt"])
>>> print(schema)
[{'key': 'party_a', 'desc': 'The first party', 'type': 'str'}, ...]
''')

add_chinese_doc('rag.doc_to_db.DocInfoExtractor', '''\
根据给定的字段结构（schema）从文档中抽取具体的关键信息值，返回格式为 key-value 字典。

Args:
    无
''')

add_english_doc('rag.doc_to_db.DocInfoExtractor', '''\
Extracts specific values for key fields from a document according to a provided schema. Returns a dictionary of key-value pairs.

Args:
    None
''')

add_example('rag.doc_to_db.DocInfoExtractor', '''\
>>> from lazyllm.components.doc_info_extractor import DocInfoExtractor
>>> from lazyllm import OnlineChatModule
>>> extractor = DocInfoExtractor()
>>> m = OnlineChatModule(source="openai")
>>> schema = [{"key": "party_a", "desc": "Party A name", "type": "str"}]
>>> info = extractor.extract_doc_info(m, "contract.txt", schema)
>>> print(info)
{'party_a': 'ABC Corp'}
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor', '''\
用于将文档信息抽取并导出到数据库中。

该类通过分析文档主题、抽取字段结构、从文档中提取关键信息，并将其保存至数据库表中。

Args:
    sql_manager (SqlManager): 数据库管理模块。
    doc_table_name (str): 存储文档字段的数据库表名，默认为`lazyllm_doc_elements`。

Note:
    - 如果表已存在，会自动检测并避免重复创建。
    - 如果你希望重置字段结构，使用 `reset_doc_info_schema` 方法。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor', '''\
Used to extract information from documents and export it to a database.

This class analyzes document topics, extracts schema structure, pulls out key information, and saves it into a database table.

Args:
    sql_manager (SqlManager): The SQL management module.
    doc_table_name (str): The table name to store document fields. Default is ``lazyllm_doc_elements``.

Note:
    - If the table already exists, it checks and avoids redundant creation.
    - Use `reset_doc_info_schema` to reset the schema if necessary.
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.extract_info_from_docs', '''\
从文档中提取结构化数据库信息。

该函数使用嵌入和检索技术，在提供的文档中获取数据库相关的文本片段，用于后续模式生成。

Args:
    docs (list[DocNode]): 输入文档列表。
    num_nodes (int): 要提取的片段数量，默认为10。

Returns:
    list[DocNode]: 提取出的相关文档片段。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor.extract_info_from_docs', '''\
Extract structured database-related information from documents.

This function uses embedding and retrieval techniques to identify relevant text fragments in the provided documents for schema generation.

Args:
    docs (list[DocNode]): List of input documents.
    num_nodes (int): Number of text fragments to retrieve. Default is 10.

Returns:
    list[DocNode]: The relevant extracted document nodes.
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.analyze_info_schema_by_llm', '''\
使用大语言模型从文档节点中推断数据库信息结构。

Args:
    nodes (list[DocNode]): 文档节点列表。

Returns:
    dict: 结构化信息模式，包含表名、字段、关系等信息。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor.analyze_info_schema_by_llm', '''\
Infer structured database information using a large language model from document nodes.

Args:
    nodes (list[DocNode]): List of document nodes.

Returns:
    dict: The inferred database schema, including table names, fields, and relationships.
''')


add_chinese_doc('rag.doc_to_db.extract_db_schema_from_files', '''\
给定文档路径和LLM模型，提取文档结构信息。

Args:
    file_paths (List[str]): 要分析的文档路径。
    llm (Union[OnlineChatModule, TrainableModule]): 支持聊天的模型模块。

Returns:
    DocInfoSchema: 提取出的字段结构描述。
''')

add_english_doc('rag.doc_to_db.extract_db_schema_from_files', '''\
Extract the schema information from documents using a given LLM.

Args:
    file_paths (List[str]): Paths of the documents to analyze.
    llm (Union[OnlineChatModule, TrainableModule]): A chat-supported LLM module.

Returns:
    DocInfoSchema: The extracted field structure schema.
''')

add_example('rag.doc_to_db.extract_db_schema_from_files', '''\
>>> import lazyllm
>>> from lazyllm.components.document_to_db import extract_db_schema_from_files
>>> llm = lazyllm.OnlineChatModule()
>>> file_paths = ["doc1.pdf", "doc2.pdf"]
>>> schema = extract_db_schema_from_files(file_paths, llm)
>>> print(schema)
''')

add_chinese_doc('rag.readers.DocxReader', """\
docx格式文件解析器，从 `.docx` 文件中读取文本内容并封装为文档节点（DocNode）列表。

Args:
    file (Path): `.docx` 文件路径。
    fs (Optional[AbstractFileSystem]): 可选的文件系统对象，支持自定义读取方式。

Returns:
    List[DocNode]: 包含文档中所有文本内容的节点列表。
""")

add_english_doc('rag.readers.DocxReader', """\
A docx format file parser, reading text content from a `.docx` file and return a list of `DocNode` objects.

Args:
    file (Path): Path to the `.docx` file.
    fs (Optional[AbstractFileSystem]): Optional file system object for custom reading.

Returns:
    List[DocNode]: A list containing the extracted text content as `DocNode` instances.
""")

add_chinese_doc('rag.readers.EpubReader', """\
用于读取 `.epub` 格式电子书的文件读取器。

继承自 `LazyLLMReaderBase`，只需实现 `_load_data` 方法，即可通过 `Document` 组件自动加载 `.epub` 文件中的内容。

注意：当前版本不支持通过 fsspec 文件系统（如远程路径）加载 epub 文件，若提供 `fs` 参数，将回退到本地文件读取。

Returns:
    List[DocNode]: 所有章节内容合并后的文本节点列表。
""")

add_english_doc('rag.readers.EpubReader', """\
A file reader for `.epub` format eBooks.

Inherits from `LazyLLMReaderBase`, and only needs to implement `_load_data`. The `Document` module can automatically use this class to load `.epub` files.

Note: Reading from fsspec file systems (e.g., remote paths) is not supported in this version. If `fs` is specified, it will fall back to reading from the local file system.

Returns:
    List[DocNode]: A single node containing all merged chapter content from the EPUB file.
""")

add_chinese_doc('rag.readers.HWPReader', '''\
HWP文件解析器，支持从本地文件系统读取 HWP 文件。它会从文档中提取正文部分的文本内容，返回 DocNode 列表。

HWP 是一种专有的二进制格式，主要在韩国使用。由于格式封闭，因此只能解析部分内容（如文本段落），但对常规文本提取已经足够使用。

Args:
    return_trace (bool): 是否启用 trace 日志记录，默认为 ``True``。
''')

add_english_doc('rag.readers.HWPReader', '''
A HWP format file parser. It supports loading from the local filesystem. It extracts body text from the `.hwp` file and returns it as a list of DocNode objects.

HWP is a proprietary binary document format used primarily in Korea. This reader focuses on extracting the plain text from the body sections of the document.

Args:
    return_trace (bool): Whether to enable trace logging. Defaults to ``True``.
''')

add_chinese_doc('rag.readers.ImageReader', '''\
用于从图片文件中读取内容的模块。支持保留图片、解析图片中的文本（基于OCR或预训练视觉模型），并返回文本和图片路径的节点列表。

Args:
    parser_config (Optional[Dict]): 解析器配置，包含模型和处理器，默认为 None。当设置 parse_text=True 且 parser_config=None 时，会自动根据 text_type 加载相应模型。
    keep_image (bool): 是否保留图片的 base64 编码，默认为 False。
    parse_text (bool): 是否解析图片中的文本，默认为 False。
    text_type (str): 解析文本的类型，支持 ``text``（默认）和 ``plain_text``。当为 ``plain_text`` 时，使用 pytesseract 进行OCR；否则使用预训练视觉编码解码模型。
    pytesseract_model_kwargs (Optional[Dict]): 传递给 pytesseract OCR 的可选参数，默认为空字典。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.ImageReader', '''\
Module for reading content from image files. Supports keeping the image as base64, parsing text from images using OCR or pretrained vision models, and returns a list of nodes with text and image path.

Args:
    parser_config (Optional[Dict]): Parser configuration containing the model and processor. Defaults to None. When parse_text=True and parser_config is None, relevant models will be auto-loaded based on text_type.
    keep_image (bool): Whether to keep the image as base64 string. Default is False.
    parse_text (bool): Whether to parse text from the image. Default is False.
    text_type (str): Type of text parsing. Supports ``text`` (default) and ``plain_text``. If ``plain_text``, pytesseract OCR is used; otherwise a pretrained vision encoder-decoder model is used.
    pytesseract_model_kwargs (Optional[Dict]): Optional arguments passed to pytesseract OCR. Defaults to empty dict.
    return_trace (bool): Whether to record the processing trace. Default is True.
''')

add_chinese_doc('rag.readers.IPYNBReader', '''\
用于读取和解析 Jupyter Notebook (.ipynb) 文件的模块。将 notebook 转换成脚本文本后，按代码单元划分为多个文档节点，或合并为单一文本节点。

Args:
    parser_config (Optional[Dict]): 预留的解析器配置参数，当前未使用，默认为 None。
    concatenate (bool): 是否将所有代码单元合并成一个整体文本节点，默认为 False，即分割为多个节点。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.IPYNBReader', '''\
Module for reading and parsing Jupyter Notebook (.ipynb) files. Converts the notebook to script text, then splits it by code cells into multiple document nodes or concatenates into a single text node.

Args:
    parser_config (Optional[Dict]): Reserved parser configuration parameter, currently unused. Defaults to None.
    concatenate (bool): Whether to concatenate all code cells into one text node. Defaults to False (split into multiple nodes).
    return_trace (bool): Whether to record processing trace. Default is True.
''')

add_chinese_doc('rag.readers.MagicPDFReader', '''\
用于通过 MagicPDF 服务解析 PDF 文件内容的模块。支持上传文件或通过 URL 方式调用解析接口，解析结果经过回调函数处理成文档节点列表。

Args:
    magic_url (str): MagicPDF 服务的接口 URL。
    callback (Optional[Callable[[List[dict], Path, dict], List[DocNode]]]): 解析结果回调函数，接收解析元素列表、文件路径及额外信息，返回文档节点列表。默认将所有文本合并为一个节点。
    upload_mode (bool): 是否采用文件上传模式调用接口，默认为 False，即通过 JSON 请求文件路径。
''')

add_english_doc('rag.readers.MagicPDFReader', '''\
Module to parse PDF content via the MagicPDF service. Supports file upload or URL-based parsing, with a callback to process the parsed elements into document nodes.

Args:
    magic_url (str): The MagicPDF service API URL.
    callback (Optional[Callable[[List[dict], Path, dict], List[DocNode]]]): A callback function that takes parsed element list, file path, and extra info, returns a list of DocNode. Defaults to merging all text into a single node.
    upload_mode (bool): Whether to use file upload mode for the API call. Default is False, meaning JSON request with file path.
''')

add_chinese_doc('rag.readers.MarkdownReader', '''\
用于读取和解析 Markdown 文件的模块。支持去除超链接和图片，按标题和内容将 Markdown 划分成若干文本段落节点。

Args:
    remove_hyperlinks (bool): 是否移除超链接，默认 True。
    remove_images (bool): 是否移除图片标记，默认 True。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.MarkdownReader', '''\
Module for reading and parsing Markdown files. Supports removing hyperlinks and images, and splits Markdown into text segments by headers, returning document nodes.

Args:
    remove_hyperlinks (bool): Whether to remove hyperlinks, default is True.
    remove_images (bool): Whether to remove image tags, default is True.
    return_trace (bool): Whether to record processing trace, default is True.
''')

add_chinese_doc('rag.readers.MarkdownReader.remove_images', '''\
移除内容中形如 ![[...]] 的自定义图片标签。

Args:
    content (str): 输入的 markdown 内容。

Returns:
    str: 移除图片标签后的内容。
''')

add_english_doc('rag.readers.MarkdownReader.remove_images', '''\
Remove custom image tags of the form ![[...]] from the content.

Args:
    content (str): Input markdown content.

Returns:
    str: Content with image tags removed.
''')

add_chinese_doc('rag.readers.MarkdownReader.remove_hyperlinks', '''\
移除 Markdown 超链接，将 [文本](链接) 转换为纯文本。

Args:
    content (str): 输入的 markdown 内容。

Returns:
    str: 移除超链接后的内容，仅保留链接文本。
''')

add_english_doc('rag.readers.MarkdownReader.remove_hyperlinks', '''\
Remove markdown hyperlinks, converting [text](url) to just text.

Args:
    content (str): Input markdown content.

Returns:
    str: Content with hyperlinks removed, only link text retained.
''')

add_chinese_doc('rag.readers.MboxReader', '''\
用于解析 Mbox 邮件存档文件的模块。读取邮件内容并格式化为文本，支持限制最大邮件数和自定义消息格式。

Args:
    max_count (int): 最大读取的邮件数量，默认 0 表示读取全部邮件。
    message_format (str): 邮件文本格式模板，支持使用 ``{_date}``、``{_from}``、``{_to}``、``{_subject}`` 和 ``{_content}`` 占位符。
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.MboxReader', '''\
Module to parse Mbox email archive files. Reads email messages and formats them into text. Supports limiting the maximum number of messages and custom message formatting.

Args:
    max_count (int): Maximum number of emails to read. Default 0 means read all.
    message_format (str): Template string for formatting each message, supports placeholders ``{_date}``, ``{_from}``, ``{_to}``, ``{_subject}``, and ``{_content}``.
    return_trace (bool): Whether to record processing trace. Default is True.
''')


add_english_doc('rag.store.ChromadbStore', '''
Inherits from the abstract base class StoreBase. This class is mainly used to store and manage document nodes (DocNode), supporting operations such as node addition, deletion, modification, query, index management, and persistent storage.
Args:
    group_embed_keys (Dict[str, Set[str]]): Specifies the embedding fields associated with each document group.
    embed (Dict[str, Callable]): A dictionary of embedding generation functions, supporting multiple embedding sources.
    embed_dims (Dict[str, int]): The embedding dimensions corresponding to each embedding type.
    dir (str): Path to the chromadb persistent storage directory.
    kwargs (Dict): Additional optional parameters passed to the parent class or internal components.
''')


add_chinese_doc('rag.store.ChromadbStore', '''
继承自 StoreBase 抽象基类。它主要用于存储和管理文档节点(DocNode)，支持节点增删改查、索引管理和持久化存储。
Args:
     group_embed_keys (Dict[str, Set[str]]): 指定每个文档分组所对应的嵌入字段。
    embed (Dict[str, Callable]): 嵌入生成函数或其映射，支持多嵌入源。
    embed_dims (Dict[str, int]): 每种嵌入类型对应的维度。
    dir (str): chromadb 数据库存储路径。
    kwargs (Dict): 其他可选参数，传递给父类或内部组件。
''')

add_example('rag.store.ChromadbStore', '''
>>> from lazyllm.tools.rag.chroma_store import ChromadbStore
>>> from typing import Dict, List
>>> import numpy as np
>>> store = ChromadbStore(
...     group_embed_keys={"articles": {"title_embed", "content_embed"}},
...     embed={
...         "title_embed": lambda x: np.random.rand(128).tolist(),
...         "content_embed": lambda x: np.random.rand(256).tolist()
...     },
...     embed_dims={"title_embed": 128, "content_embed": 256},
...     dir="./chroma_data"
... )
>>> store.update_nodes([node1, node2])
>>> results = store.query(query_text="文档内容", group_name="articles", top_k=2)
>>> for node in results:
...     print(f"找到文档: {node._content[:20]}...")
>>> store.remove_nodes(doc_ids=["doc1"])
''')

add_english_doc('rag.store.ChromadbStore.update_nodes', '''
Update a group of DocNode objects.
Args:
    nodes (DocNode): The list of DocNode objects to be updated.
''')


add_chinese_doc('rag.store.ChromadbStore.update_nodes', '''
更新一组 DocNode 节点。
Args:
    nodes(DocNode): 需要更新的 DocNode 列表。
''')


add_english_doc('rag.store.ChromadbStore.remove_nodes', '''
Delete nodes based on specified conditions.
Args:
    doc_ids (str): Delete by document ID.
    group_name (str): Specify the group name for deletion.
    uids (str): Delete by unique node ID.
''')


add_chinese_doc('rag.store.ChromadbStore.remove_nodes', '''
删除指定条件的节点。
Args:
    doc_ids(str): 按文档 ID 删除。
    group_name(str): 限定删除的组名。
    uids(str): 按节点唯一 ID 删除。
''')


add_english_doc('rag.store.ChromadbStore.update_doc_meta', '''
Update the metadata of a document.
Args:
    doc_id (str): The ID of the document to be updated.
    metadata (dict): The new metadata (key-value pairs).
''')


add_chinese_doc('rag.store.ChromadbStore.update_doc_meta', '''
更新文档的元数据。。
Args:
    doc_id(str):需要更新的文档 ID。
    metadata(dict):新的元数据（键值对）。
''')


add_english_doc('rag.store.ChromadbStore.get_nodes', '''
Query nodes based on specified conditions.
Args:
    group_name (str): The name of the group to which the nodes belong.
    uids (List[str]): A list of unique node IDs.
    doc_ids (Set[str]): A set of document IDs.
    **kwargs: Additional optional parameters.
''')


add_chinese_doc('rag.store.ChromadbStore.get_nodes', '''
根据条件查询节点。
Args:
    group_name(str]):节点所属的组名。
    uids(List[str]):节点唯一 ID 列表。
    doc_ids	(Set[str])：文档 ID 集合。
    **kwargs:其他扩展参数。
''')


add_english_doc('rag.store.ChromadbStore.activate_group', '''
Activate the specified group.
Args:
    group_names([str, List[str]]): Activate by group name.
''')


add_chinese_doc('rag.store.ChromadbStore.activate_group', '''
激活指定的组。
Args:
    group_names([str, List[str]])：按组名激活。
''')

add_english_doc('rag.store.ChromadbStore.activated_groups', '''
Activate groups. Return the list of currently activated group names.
''')


add_chinese_doc('rag.store.ChromadbStore.activated_groups', '''
激活组，返回当前激活的组名列表。
''')
add_english_doc('rag.store.ChromadbStore.query', '''
Execute a query using the default index.
Args:
    args: Query parameters.
    kwargs: Additional optional parameters.
''')


add_chinese_doc('rag.store.ChromadbStore.query', '''
通过默认索引执行查询。
Args:
    args：查询参数。
    kwargs：其他扩展参数。
''')

add_english_doc('rag.store.ChromadbStore.is_group_active', '''
Check whether the specified group is active.
Args:
    name (str): The name of the group.
''')

add_chinese_doc('rag.store.ChromadbStore.is_group_active', '''
检查指定组是否激活。
Args:
    name(str)：组名。
''')


add_english_doc('rag.store.ChromadbStore.all_groups', '''
Return the list of all group names.
''')


add_chinese_doc('rag.store.ChromadbStore.all_groups', '''
返回所有组名列表。
''')

add_english_doc('rag.store.ChromadbStore.register_index', '''
Register a custom index.
Args:
    type (str): The name of the index type.
    index (IndexBase): An object implementing the IndexBase interface.
''')


add_chinese_doc('rag.store.ChromadbStore.register_index', '''
注册自定义索引。
Args:
    type(str):索引类型名称。
    index(IndexBase):实现 IndexBase 的对象。
''')


add_english_doc('rag.store.ChromadbStore.get_index', '''
Get the index of the specified type.
Args:
    type (str): The type of the index.
''')


add_chinese_doc('rag.store.ChromadbStore.get_index', '''
获取指定类型的索引。
Args:
    type(str):索引类型
''')


add_english_doc('rag.store.ChromadbStore.clear_cache', '''
Clear the ChromaDB collections and memory cache for specified groups or all groups.
Args:
    group_names (List[str]): List of group names. If None, clear all groups.
''')


add_chinese_doc('rag.store.ChromadbStore.clear_cache', '''
清除指定组或所有组的 ChromaDB 集合和内存缓存。
Args:
    group_names(List[str])：组名列表，为 None 时清除所有组。
''')





add_english_doc('rag.store.MilvusStore', '''
Inherits from the StoreBase abstract base class. Implements a vector database based on Milvus. Its functionality is similar to ChromadbStore, used for storing, managing, indexing, and querying embedded document nodes (DocNode).
Args:
    group_embed_keys (Dict[str, Set[str]]): Specifies the embedding fields for each group.
    embed (Dict[str, Callable]): Embedding functions for each field.
    embed_dims (Dict[str, int]): Vector dimensions for each embedding field.
    embed_datatypes (Dict[str, DataType]): Vector types for each embedding field (must comply with Milvus types).
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Description of global metadata fields, used to configure other non-vector fields in Milvus.
    url (str): Milvus connection address, supporting local or remote connections.
    index_kwargs (Union[Dict, List]): Optional index parameters for creating Milvus vector indexes, such as IVF, HNSW parameters.
    db_name (str): Optional, defaults to "lazyllm". Represents the database name in Milvus.
''')


add_chinese_doc('rag.store.MilvusStore', '''
继承自 StoreBase 抽象基类。基于 Milvus 向量数据库实现。其功能和 ChromadbStore 类似, 用于存储、管理、索引和查询嵌入向量化后的文档节点(DocNode)。
Args:
    group_embed_keys (Dict[str, Set[str]]): 指定每个group所对应的嵌入字段。
    embed (Dict[str, Callable]): 每种字段对应的 embedding 函数.
    embed_dims (Dict[str, int]): 每个嵌入字段的向量维度。
    embed_datatypes(Dict[str, DataType]): 每个嵌入字段的向量类型（需符合 Milvus 类型）。
    global_metadata_descDict([str, GlobalMetadataDesc])：全局元数据字段的说明，用于配置 Milvus 中的其他非向量字段。
    url(str):Milvus 的连接地址，支持本地或远程。
    index_kwargs:([Union[Dict, List]]):可选的索引参数，用于创建 Milvus 的向量索引，例如 IVF、HNSW 参数。
    db_name(str):可选，默认 "lazyllm"。表示 Milvus 中的数据库名。
''')

add_example('rag.store.MilvusStore', '''
>>> from lazyllm.tools.rag.milvus_store import MilvusStore
>>> from typing import Dict, List
>>> import numpy as np
>>> store = MilvusStore(
...     group_embed_keys={
...         "articles": {"text"},
...         "faqs": {"question"}
...     },
...     embed={
...         "text": lambda x: np.random.rand(128).tolist(),
...         "question": lambda x: np.random.rand(128).tolist()
...     },
...     embed_dims={"text": 128, "question": 128},
...     embed_datatypes={"text": DataType.FLOAT_VECTOR, "question": DataType.FLOAT_VECTOR},
...     global_metadata_desc=None,
...     uri="http://localhost:19530",
...     index_kwargs={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
...     db_name="test_db"
... )
>>> store.update_nodes([node1, node2])
>>> results = store.query(query_text="文档内容", group_name="articles", top_k=2)
>>> for node in results:
...     print(f"找到文档: {node._content[:20]}...")
>>> store.remove_nodes(doc_ids=["doc1"])
''')

add_english_doc('rag.store.MilvusStore.update_nodes', '''
Update or insert nodes into Milvus collections and memory store.
Args:
    nodes (List[DocNode]): List of document nodes to update.
''')

add_chinese_doc('rag.store.MilvusStore.update_nodes', '''
更新或插入节点到 Milvus 集合和内存存储中。
Args:
    nodes (List[DocNode]): 需要更新的文档节点列表。
''')

add_english_doc('rag.store.MilvusStore.update_doc_meta', '''
Update metadata for a document and sync to all related nodes.
Args:
    doc_id (str): Target document ID.
    metadata (dict): New metadata key-value pairs.
''')

add_chinese_doc('rag.store.MilvusStore.update_doc_meta', '''
更新文档元数据并同步到所有关联节点。
Args:
    doc_id (str): 目标文档ID。
    metadata (dict): 新的元数据键值对。
''')

add_english_doc('rag.store.MilvusStore.remove_nodes', '''
Remove nodes by document IDs, group name, or node UIDs.
Args:
    doc_ids (Optional[List[str]]): Document IDs filter.
    group_name (Optional[str]): Group name filter.
    uids (Optional[List[str]]): Node UIDs filter.
''')

add_chinese_doc('rag.store.MilvusStore.remove_nodes', '''
通过文档ID、组名或节点UID删除节点。
Args:
    doc_ids (Optional[List[str]]): 文档ID过滤条件。
    group_name (Optional[str]): 组名过滤条件。
    uids (Optional[List[str]]): 节点UID过滤条件。
''')
add_english_doc('rag.store.MilvusStore.get_nodes', '''
Query nodes with flexible filtering options.
Args:
    group_name (Optional[str]): Group name filter.
    uids (Optional[List[str]]): Node UIDs filter.
    doc_ids (Optional[Set[str]]): Document IDs filter.
    **kwargs: Additional query parameters.
Returns:
    List[DocNode]: Matched document nodes.
''')

add_chinese_doc('rag.store.MilvusStore.get_nodes', '''
通过多条件查询节点。
Args:
    group_name (Optional[str]): 组名过滤条件。
    uids (Optional[List[str]]): 节点UID过滤条件。
    doc_ids (Optional[Set[str]]): 文档ID过滤条件。
    **kwargs: 其他查询参数。
Returns:
    List[DocNode]: 匹配的文档节点列表。
''')

add_english_doc('rag.store.MilvusStore.query', '''
Semantic search with vector similarity.
Args:
    query (str): Query text.
    group_name (str): Target group name.
    similarity_cut_off (Optional[Union[float, Dict[str, float]]]): Similarity threshold.
    topk (int): Number of results to return.
    embed_keys (List[str]): Embedding keys for search.
    filters (Optional[Dict]): Metadata filters.
Returns:
    List[DocNode]: Nodes with similarity scores.
''')

add_chinese_doc('rag.store.MilvusStore.query', '''
基于向量相似度的语义搜索。
Args:
    query (str): 查询文本。
    group_name (str): 目标组名。
    similarity_cut_off (Optional[Union[float, Dict[str, float]]): 相似度阈值。
    topk (int): 返回结果数量。
    embed_keys (List[str]): 用于搜索的嵌入键。
    filters (Optional[Dict]): 元数据过滤条件。
Returns:
    List[DocNode]: 带相似度分数的节点列表。
''')

add_english_doc('rag.store.MilvusStore.activate_group', '''
Activate one or multiple groups for operations.
Args:
    group_names (Union[str, List[str]]): Group name(s) to activate.
''')

add_chinese_doc('rag.store.MilvusStore.activate_group', '''
激活一个或多个组用于后续操作。
Args:
    group_names (Union[str, List[str]]): 要激活的组名（单个或列表）。
''')

add_english_doc('rag.store.MilvusStore.get_index', '''
Get index instance by type.
Args:
    type (Optional[str]): Index type name, defaults to "default".
''')

add_chinese_doc('rag.store.MilvusStore.get_index', '''
获取指定类型的索引实例。
Args:
    type (Optional[str]): 索引类型名称，默认为"default"。
''')

add_english_doc('rag.store.MilvusStore.register_index', '''
Register custom index type.
Args:
    type (str): Index type name.
    index (IndexBase): Custom index instance.
''')

add_chinese_doc('rag.store.MilvusStore.register_index', '''
注册自定义索引类型。
Args:
    type (str): 索引类型名称。
    index (IndexBase): 自定义索引实例。
''')

add_english_doc('rag.store.MilvusStore.activated_groups', '''
Get names of all activated groups.
Returns:
    List[str]: Active group names.
''')

add_chinese_doc('rag.store.MilvusStore.activated_groups', '''
获取所有已激活的组名。
Returns:
    List[str]: 活跃组名列表。
''')

add_english_doc('rag.store.MilvusStore.is_group_active', '''
Check if a group is activated.
Args:
    name (str): Group name to check.
''')

add_chinese_doc('rag.store.MilvusStore.is_group_active', '''
检查指定组是否激活。
Args:
    name (str): 要检查的组名。
''')

# ---------------------------------------------------------------------------- #

# rag/rerank.py

add_english_doc('Reranker', '''\
Initializes a Rerank module for postprocessing and reranking of nodes (documents).
This constructor initializes a Reranker module that configures a reranking process based on a specified reranking type. It allows for the dynamic selection and instantiation of reranking kernels (algorithms) based on the type and provided keyword arguments.

Args:
    name: The type of reranker used for the postprocessing and reranking process. Defaults to 'ModuleReranker'.
    target (str): **Deprecated** parameter, only used to notify users.
    output_format: Specifies the output format. Defaults to None. Optional values include 'content' and 'dict'. 
        - 'content' means the output is in string format.
        - 'dict' means the output is a dictionary.
    join: Determines whether to join the top-k output nodes.
        - When `output_format` is 'content':
            - If set to True, returns a single long string.
            - If set to False, returns a list of strings, each representing one node’s content.
        - When `output_format` is 'dict':
            - Joining is not supported; `join` defaults to False.
            - Returns a dictionary with three keys: 'content', 'embedding', and 'metadata'.
    kwargs: Additional keyword arguments passed to the reranker upon instantiation.
    name: The type of reranker used for the postprocessing and reranking process. Defaults to 'ModuleReranker'.
    target (str): **Deprecated** parameter, only used to notify users.
    output_format: Specifies the output format. Defaults to None. Optional values include 'content' and 'dict'. 
        - 'content' means the output is in string format.
        - 'dict' means the output is a dictionary.
    join: Determines whether to join the top-k output nodes.
        - When `output_format` is 'content':
            - If set to True, returns a single long string.
            - If set to False, returns a list of strings, each representing one node’s content.
        - When `output_format` is 'dict':
            - Joining is not supported; `join` defaults to False.
            - Returns a dictionary with three keys: 'content', 'embedding', and 'metadata'.
    kwargs: Additional keyword arguments passed to the reranker upon instantiation.
**Detailed explanation of reranker types**

- Reranker: Instantiates a `SentenceTransformerRerank` reranker with a list of document nodes and a query.\n
- Reranker: Instantiates a `SentenceTransformerRerank` reranker with a list of document nodes and a query.\n
- KeywordFilter: This registered reranking function instantiates a KeywordNodePostprocessor with specified required and excluded keywords. It filters nodes based on the presence or absence of these keywords.
''')

add_chinese_doc('Reranker', '''\
用于创建节点（文档）后处理和重排序的模块。

Args:
    name: 用于后处理和重排序过程的排序器类型。默认为 'ModuleReranker'。
    target(str):已废弃参数，仅用于提示用户。
    output_format: 代表输出格式，默认为None，可选值有 'content' 和 'dict'，其中 content 对应输出格式为字符串，dict 对应字典。
    join: 是否联合输出的 k 个节点，当输出格式为 content 时，如果设置该值为 True，则输出一个长字符串，如果设置为 False 则输出一个字符串列表，其中每个字符串对应每个节点的文本内容。当输出格式是 dict 时，不能联合输出，此时join默认为False,，将输出一个字典，包括'content、'embedding'、'metadata'三个key。
    name: 用于后处理和重排序过程的排序器类型。默认为 'ModuleReranker'。
    target(str):已废弃参数，仅用于提示用户。
    output_format: 代表输出格式，默认为None，可选值有 'content' 和 'dict'，其中 content 对应输出格式为字符串，dict 对应字典。
    join: 是否联合输出的 k 个节点，当输出格式为 content 时，如果设置该值为 True，则输出一个长字符串，如果设置为 False 则输出一个字符串列表，其中每个字符串对应每个节点的文本内容。当输出格式是 dict 时，不能联合输出，此时join默认为False,，将输出一个字典，包括'content、'embedding'、'metadata'三个key。
    kwargs: 传递给重新排序器实例化的其他关键字参数。

详细解释排序器类型

  - Reranker: 实例化一个具有待排序的文档节点node列表和 query的 SentenceTransformerRerank 重排序器。
  - Reranker: 实例化一个具有待排序的文档节点node列表和 query的 SentenceTransformerRerank 重排序器。
  - KeywordFilter: 实例化一个具有指定必需和排除关键字的 KeywordNodePostprocessor。它根据这些关键字的存在或缺失来过滤节点。
''')

add_example('Reranker', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, Reranker, Retriever, DocNode
>>> from lazyllm.tools import Document, Reranker, Retriever, DocNode
>>> m = lazyllm.OnlineEmbeddingModule()
>>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> retriever = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
>>> reranker = Reranker(DocNode(text=user_data),query="user query")
>>> reranker = Reranker(DocNode(text=user_data),query="user query")
>>> ppl = lazyllm.ActionModule(retriever, reranker)
>>> ppl.start()
>>> print(ppl("user query"))
''')

add_english_doc('Reranker.register_reranker', '''\
A class decorator factory method that provides a flexible mechanism for registering custom reranking algorithms to the `Reranker` class.
Args:
    func (Optional[Callable]): The reranking function or class to register. This can be omitted when using decorator syntax (@).
    batch (bool): Whether to process nodes in batches. Defaults to False, meaning nodes are processed individually.
''')


add_chinese_doc('Reranker.register_reranker', '''\
是一个类装饰器工厂方法，它的核心作用是为 Reranker 类提供灵活的排序算法注册机制
Args:
    func (Optional[Callable]):  要注册的排序函数或排序器类。当使用装饰器语法(@)时可省略。
    batch (bool):是否批量处理节点。默认为False，表示逐节点处理。
''')

add_example('Reranker.register_reranker', '''
@Reranker.register_reranker
def my_reranker(node: DocNode, **kwargs):
    return node.score * 0.8  # 自定义分数计算
''')

add_english_doc('Reranker.register_reranker', '''\
A class decorator factory method that provides a flexible mechanism for registering custom reranking algorithms to the `Reranker` class.
Args:
    func (Optional[Callable]): The reranking function or class to register. This can be omitted when using decorator syntax (@).
    batch (bool): Whether to process nodes in batches. Defaults to False, meaning nodes are processed individually.
''')


add_chinese_doc('Reranker.register_reranker', '''\
是一个类装饰器工厂方法，它的核心作用是为 Reranker 类提供灵活的排序算法注册机制
Args:
    func (Optional[Callable]):  要注册的排序函数或排序器类。当使用装饰器语法(@)时可省略。
    batch (bool):是否批量处理节点。默认为False，表示逐节点处理。
''')

add_example('Reranker.register_reranker', '''
@Reranker.register_reranker
def my_reranker(node: DocNode, **kwargs):
    return node.score * 0.8  # 自定义分数计算
''')

# ---------------------------------------------------------------------------- #

# rag/retriever.py

add_english_doc('Retriever', '''
Create a retrieval module for document querying and retrieval. This constructor initializes a retrieval module that configures the document retrieval process based on the specified similarity metric.

Args:
    doc: An instance of the document module. The document module can be a single instance or a list of instances. If it is a single instance, it means searching for a single Document, and if it is a list of instances, it means searching for multiple Documents.
    group_name: The name of the node group on which to perform the retrieval.
    similarity: The similarity function to use for setting up document retrieval. Defaults to 'dummy'. Candidates include ["bm25", "bm25_chinese", "cosine"].
    similarity_cut_off: Discard the document when the similarity is below the specified value. In a multi-embedding scenario, if you need to specify different values for different embeddings, you need to specify them in a dictionary, where the key indicates which embedding is specified and the value indicates the corresponding threshold. If all embeddings use the same threshold, you only need to specify one value.
    index: The type of index to use for document retrieval. Currently, only 'default' is supported.
    topk: The number of documents to retrieve with the highest similarity.
    embed_keys: Indicates which embeddings are used for retrieval. If not specified, all embeddings are used for retrieval.
    target:The name of the target document group for result conversion
    target:The name of the target document group for result conversion
    output_format: Represents the output format, with a default value of None. Optional values include 'content' and 'dict', where 'content' corresponds to a string output format and 'dict' corresponds to a dictionary.
    join:  Determines whether to concatenate the output of k nodes - when output format is 'content', setting True returns a single concatenated string while False returns a list of strings (each corresponding to a node's text content); when output format is 'dict', joining is unsupported (join defaults to False) and the output will be a dictionary containing 'content', 'embedding' and 'metadata' keys.


The `group_name` has three built-in splitting strategies, all of which use `SentenceSplitter` for splitting, with the difference being in the chunk size:

- CoarseChunk: Chunk size is 1024, with an overlap length of 100
- MediumChunk: Chunk size is 256, with an overlap length of 25
- FineChunk: Chunk size is 128, with an overlap length of 12

Also, `Image` is available for `group_name` since LazyLLM supports image embedding and retrieval.
''')

add_chinese_doc('Retriever', '''
创建一个用于文档查询和检索的检索模块。此构造函数初始化一个检索模块，该模块根据指定的相似度度量配置文档检索过程。

Args:
    doc: 文档模块实例。该文档模块可以是单个实例，也可以是一个实例的列表。如果是单个实例，表示对单个Document进行检索，如果是实例的列表，则表示对多个Document进行检索。
    group_name: 在哪个 node group 上进行检索。
    similarity: 用于设置文档检索的相似度函数。默认为 'dummy'。候选集包括 ["bm25", "bm25_chinese", "cosine"]。
    similarity_cut_off: 当相似度低于指定值时丢弃该文档。在多 embedding 场景下，如果需要对不同的 embedding 指定不同的值，则需要使用字典的方式指定，key 表示指定的是哪个 embedding，value 表示相应的阈值。如果所有的 embedding 使用同一个阈值，则只指定一个数值即可。
    index: 用于文档检索的索引类型。目前仅支持 'default'。
    topk: 表示取相似度最高的多少篇文档。
    embed_keys: 表示通过哪些 embedding 做检索，不指定表示用全部 embedding 进行检索。
    target：目标组名，将结果转换到目标组。
    target：目标组名，将结果转换到目标组。
    output_format: 代表输出格式，默认为None，可选值有 'content' 和 'dict'，其中 content 对应输出格式为字符串，dict 对应字典。
    join: 是否联合输出的 k 个节点，当输出格式为 content 时，如果设置该值为 True，则输出一个长字符串，如果设置为 False 则输出一个字符串列表，其中每个字符串对应每个节点的文本内容。当输出格式是 dict 时，不能联合输出，此时join默认为False,，将输出一个字典，包括'content、'embedding'、'metadata'三个key。

其中 `group_name` 有三个内置的切分策略，都是使用 `SentenceSplitter` 做切分，区别在于块大小不同：

- CoarseChunk: 块大小为 1024，重合长度为 100
- MediumChunk: 块大小为 256，重合长度为 25
- FineChunk: 块大小为 128，重合长度为 12

此外，LazyLLM提供了内置的`Image`节点组存储了所有图像节点，支持图像嵌入和检索。
''')

add_example('Retriever', '''
>>> import lazyllm
>>> from lazyllm.tools import Retriever, Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule()
>>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> rm = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
>>> rm.start()
>>> print(rm("user query"))
>>> m1 = lazyllm.TrainableModule('bge-large-zh-v1.5').start()
>>> document1 = Document(dataset_path='/path/to/user/data', embed={'online':m , 'local': m1}, manager=False)
>>> document1.create_node_group(name='sentences', transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> retriever = Retriever(document1, group_name='sentences', similarity='cosine', similarity_cut_off=0.4, embed_keys=['local'], topk=3)
>>> print(retriever("user query"))
>>> document2 = Document(dataset_path='/path/to/user/data', embed={'online':m , 'local': m1}, manager=False)
>>> document2.create_node_group(name='sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
>>> retriever2 = Retriever([document1, document2], group_name='sentences', similarity='cosine', similarity_cut_off=0.4, embed_keys=['local'], topk=3)
>>> print(retriever2("user query"))
>>>
>>> filters = {
>>>     "author": ["A", "B", "C"],
>>>     "public_year": [2002, 2003, 2004],
>>> }
>>> document3 = Document(dataset_path='/path/to/user/data', embed={'online':m , 'local': m1}, manager=False)
>>> document3.create_node_group(name='sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
>>> retriever3 = Retriever([document1, document3], group_name='sentences', similarity='cosine', similarity_cut_off=0.4, embed_keys=['local'], topk=3)
>>> print(retriever3(query="user query", filters=filters))
>>> document4 = Document(dataset_path='/path/to/user/data', embed=lazyllm.TrainableModule('siglip'))
>>> retriever4 = Retriever(document4, group_name='Image', similarity='cosine')
>>> nodes = retriever4("user query")
>>> print([node.get_content() for node in nodes])
>>> document5 = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> rm = Retriever(document5, group_name='CoarseChunk', similarity='bm25_chinese', similarity_cut_off=0.01, topk=3, output_format='content')
>>> rm.start()
>>> print(rm("user query"))
>>> document6 = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> rm = Retriever(document6, group_name='CoarseChunk', similarity='bm25_chinese', similarity_cut_off=0.01, topk=3, output_format='content', join=True)
>>> rm.start()
>>> print(rm("user query"))
>>> document7 = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> rm = Retriever(document7, group_name='CoarseChunk', similarity='bm25_chinese', similarity_cut_off=0.01, topk=3, output_format='dict')
>>> rm.start()
>>> print(rm("user query"))
''')

add_english_doc('rag.retriever.TempDocRetriever', '''
A temporary document retriever that inherits from ModuleBase and _PostProcess, used for quickly processing temporary files and performing retrieval tasks.
Args:
    embed: The embedding function.
    output_format: The format of the output result (e.g., JSON). Optional, defaults to None.
    join: Whether to merge multiple result segments (set to True or specify a separator like "\n").
''')

add_chinese_doc('rag.retriever.TempDocRetriever', '''
临时文档检索器，继承自 ModuleBase 和 _PostProcess，用于快速处理临时文件并执行检索任务。
Args:
    embed:嵌入函数。
    output_format:结果输出格式(如json),可选默认为None
    join:是否合并多段结果(True或用分隔符如"\n")
''')

add_example('rag.retriever.TempDocRetriever', '''
>>> import lazyllm
>>> from lazyllm.tools import TempDocRetriever, Document, SentenceSplitter
>>> retriever = TempDocRetriever(output_format="text", join="\n---------------\n")
    retriever.create_node_group(transform=lambda text: [s.strip() for s in text.split("。") if s] )
    retriever.add_subretriever(group=Document.MediumChunk, topk=3)
    files = ["机器学习是AI的核心领域。深度学习是其重要分支。"]
    results = retriever.forward(files, "什么是机器学习?")
    print(results)
''')

add_english_doc('rag.retriever.TempDocRetriever.create_node_group', '''
Create a node group with specific processing pipeline.
Args:
    name (str): Name of the node group. Auto-generated if None.
    transform (Callable): Function to process documents in this group.
    parent (str): Parent group name. Defaults to root group.
    trans_node (bool): Whether to transform nodes. Inherits from parent if None.
    num_workers (int): Parallel workers for processing. Default 0 (sequential).
    **kwargs: Additional group parameters.
''')

add_chinese_doc('rag.retriever.TempDocRetriever.create_node_group', '''
创建具有特定处理流程的节点组。
Args:
    name (str): 节点组名称，None时自动生成。
    transform (Callable): 该组文档的处理函数。
    parent (str): 父组名称，默认为根组。
    trans_node (bool): 是否转换节点，None时继承父组设置。
    num_workers (int): 并行处理worker数，0表示串行。
    **kwargs: 其他组参数。
''')

add_english_doc('rag.retriever.TempDocRetriever.add_subretriever', '''
Add a sub-retriever with search configuration.
Args:
    group (str): Target node group name.
    **kwargs: Retriever parameters (e.g., similarity='cosine').
Returns:
    self: For method chaining.
''')

add_chinese_doc('rag.retriever.TempDocRetriever.add_subretriever', '''
添加带搜索配置的子检索器。
Args:
    group (str): 目标节点组名称。
    **kwargs: 检索器参数（如similarity='cosine'）。
Returns:
    self: 支持链式调用。
''')

add_english_doc('rag.doc_node.DocNode', '''
Execute assigned tasks on the specified document.
Args:
    uid (str): Unique identifier.
    content (Union[str, List[Any]]): Node content.
    group (str): Document group name.
    embedding (Dict[str, List[float]]): Dictionary of embedding vectors.
    parent (Union[str, "DocNode"]): Reference to the parent node.
    store: Storage representation.
    node_groups (Dict[str, Dict]): Node storage groups.
    metadata (Dict[str, Any]): Node-level metadata.
    global_metadata (Dict[str, Any]): Document-level metadata.
    text (str): Node content, mutually exclusive with content.
''')

add_chinese_doc('rag.doc_node.DocNode', '''
在指定的文档上执行设定的任务。
Args:
    uid(str): 唯一标识符。
    content(Union[str, List[Any]]):节点内容
    group(str):文档组名
    embedding(Dict[str, List[float]]):嵌入向量字典
    parent(Union[str, "DocNode"]):父节点引用
    store:存储表示
    node_groups(Dict[str, Dict]):节点存储组
    metadata(Dict[str, Any]):节点级元数据
    global_metadata(Dict[str, Any]):文档级元数据
    text(str):节点内容与content互斥
''')

add_english_doc('rag.doc_node.DocNode.get_metadata_str', '''
Get formatted metadata string.
Args:
    mode: MetadataMode.NONE returns an empty string;  
          MetadataMode.LLM filters out metadata not needed by LLM;  
          MetadataMode.EMBED filters out metadata not needed by embedding model;  
          MetadataMode.ALL returns all metadata.
''')

add_chinese_doc('rag.doc_node.DocNode.get_metadata_str', '''
获取格式化元数据字符串
Args:
    mode: MetadataMode.NONE返回空字符串；
          MetadataMode.LLM过滤排除LLM不需要的元数据；
          MetadataMode.EMBED过滤排除嵌入模型不需要的元数据；
          MetadataMode.ALL返回全部元数据。
''')
add_english_doc('rag.doc_node.DocNode.get_text', '''
Combine metadata and content.
Args:
    metadata_mode: Same as the parameter in get_metadata_str.
''')

add_chinese_doc('rag.doc_node.DocNode.get_text', '''
组合元数据和内容
Args:
    metadata_mode: 与get_metadata_str中参数一致
''')
add_english_doc('rag.doc_node.DocNode.has_missing_embedding', '''
Check for missing embedding vectors.
Args:
    embed_keys (Union[str, List[str]]): List of target keys.
''')

add_chinese_doc('rag.doc_node.DocNode.has_missing_embedding', '''
检查缺失的嵌入向量
Args:
    embed_keys(Union[str, List[str]]): 目标键列表
''')
add_english_doc('rag.doc_node.DocNode.do_embedding', '''
Execute embedding computation.
Args:
    embed (Dict[str, Callable]): Target embedding objects.
''')

add_chinese_doc('rag.doc_node.DocNode.do_embedding', '''
执行嵌入计算
Args:
    embed(Dict[str, Callable]): 目标嵌入对象
''')
add_english_doc('rag.doc_node.DocNode.check_embedding_state', '''
Block to check the embedding status and ensure that asynchronous embedding computation is completed.
Args:
    embed_key (str): List of target keys.
''')

add_chinese_doc('rag.doc_node.DocNode.check_embedding_state', '''
阻塞检查嵌入状态,确保异步嵌入计算完成
Args:
    embed_key(str): 目标键列表
''')
add_english_doc('rag.doc_node.DocNode.to_dict', '''
Convert to dictionary format
''')

add_chinese_doc('rag.doc_node.DocNode.to_dict', '''
转换为字典格式
''')
add_english_doc('rag.doc_node.DocNode.with_score', '''
Shallow copy the original node and add a semantic relevance score.
Args:
    score: Relevance score.
''')

add_chinese_doc('rag.doc_node.DocNode.with_score', '''
浅拷贝原节点并添加语义相关分数。
Args:
    score: 相关性得分
''')
add_english_doc('rag.doc_node.DocNode.with_sim_score', '''
Shallow copy the original node and add a similarity score.
Args:
    score: Similarity score.
''')

add_chinese_doc('rag.doc_node.DocNode.with_sim_score', '''
浅拷贝原节点并添加相似度分数。
Args:
    score: 相似度得分
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader', '''
A modular document directory reader that inherits from ModuleBase, supporting reading various document formats from the file system and converting them into standardized DocNode objects.
Args:
    input_dir (Optional[str]): Input directory path. Mutually exclusive with input_files.
    input_files (Optional[List]): Directly specified list of files. Mutually exclusive with input_dir.
    exclude (Optional[List]): List of file patterns to exclude.
    exclude_hidden (bool): Whether to exclude hidden files.
    recursive (bool): Whether to recursively read subdirectories.
    encoding (str): Encoding format of text files.
    required_exts (Optional[List[str]]): Whitelist of file extensions to process.
    file_extractor (Optional[Dict[str, Callable]]): Dictionary of custom file readers.
    fs (Optional[AbstractFileSystem]): Custom file system.
    metadata_genf (Optional[Callable[[str], Dict]]): Metadata generation function that takes a file path and returns a metadata dictionary.
    num_files_limit (Optional[int]): Maximum number of files to read.
    return_trace (bool): Whether to return processing trace information.
    metadatas (Optional[Dict]): Predefined global metadata dictionary.
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader', '''
模块化的文档目录读取器，继承自 ModuleBase，支持从文件系统读取多种格式的文档并转换为标准化的 DocNode 。
Args:
    input_dir (Optional[str]): 输入目录路径。与input_files二选一，不可同时指定。
    input_files (Optional[List]):直接指定的文件列表。与input_dir二选一。
    exclude (Optional[List]):需要排除的文件模式列表。
    exclude_hidden (bool): 是否排除隐藏文件。
    recursive (bool):是否递归读取子目录。
    encoding (str):文本文件的编码格式。
    required_exts (Optional[List[str]]):需要处理的文件扩展名白名单。
    file_extractor (Optional[Dict[str, Callable]]):自定义文件阅读器字典。
    fs (Optional[AbstractFileSystem]):自定义文件系统。
    metadata_genf (Optional[Callable[[str], Dict]]):元数据生成函数，接收文件路径返回元数据字典。
    num_files_limit (Optional[int]):最大读取文件数量限制。
    return_trace (bool):是否返回处理过程追踪信息。
    metadatas (Optional[Dict]):预定义的全局元数据字典。
''')

add_example('rag.dataReader.SimpleDirectoryReader', '''
>>> import lazyllm
>>> from lazyllm.tools.dataReader import SimpleDirectoryReader
>>> reader = SimpleDirectoryReader(input_dir="yourpath/",recursive=True,exclude=["*.tmp"],required_exts=[".pdf", ".docx"])
>>> documents = reader.load_data()
''')


add_english_doc('rag.dataReader.FileReader', '''
File content reader whose main function is to convert various input file formats into concatenated plain text content.
Args:
    input_files (Optional[List]): Directly specified list of input files.
''')

add_chinese_doc('rag.dataReader.FileReader', '''
文件内容读取器，主要功能是将多种格式的输入文件转换为拼接后的纯文本内容。
Args:
    input_files (Optional[List]):直接指定的文件列表。
''')

add_example('rag.dataReader.FileReader', '''
>>> import lazyllm
>>> from lazyllm.tools.dataReader import FileReader
>>> reader = FileReader()
>>> content = reader("yourpath/") 
''')

add_english_doc('rag.retriever.TempDocRetriever', '''
A temporary document retriever that inherits from ModuleBase and _PostProcess, used for quickly processing temporary files and performing retrieval tasks.
Args:
    embed: The embedding function.
    output_format: The format of the output result (e.g., JSON). Optional, defaults to None.
    join: Whether to merge multiple result segments (set to True or specify a separator like "\n").
''')

add_chinese_doc('rag.retriever.TempDocRetriever', '''
临时文档检索器，继承自 ModuleBase 和 _PostProcess，用于快速处理临时文件并执行检索任务。
Args:
    embed:嵌入函数。
    output_format:结果输出格式(如json),可选默认为None
    join:是否合并多段结果(True或用分隔符如"\n")
''')

add_example('rag.retriever.TempDocRetriever', '''
>>> import lazyllm
>>> from lazyllm.tools import TempDocRetriever, Document, SentenceSplitter
>>> retriever = TempDocRetriever(output_format="text", join="\n---------------\n")
    retriever.create_node_group(transform=lambda text: [s.strip() for s in text.split("。") if s] )
    retriever.add_subretriever(group=Document.MediumChunk, topk=3)
    files = ["机器学习是AI的核心领域。深度学习是其重要分支。"]
    results = retriever.forward(files, "什么是机器学习?")
    print(results)
''')

add_english_doc('rag.retriever.TempDocRetriever.create_node_group', '''
Create a node group with specific processing pipeline.
Args:
    name (str): Name of the node group. Auto-generated if None.
    transform (Callable): Function to process documents in this group.
    parent (str): Parent group name. Defaults to root group.
    trans_node (bool): Whether to transform nodes. Inherits from parent if None.
    num_workers (int): Parallel workers for processing. Default 0 (sequential).
    **kwargs: Additional group parameters.
''')

add_chinese_doc('rag.retriever.TempDocRetriever.create_node_group', '''
创建具有特定处理流程的节点组。
Args:
    name (str): 节点组名称，None时自动生成。
    transform (Callable): 该组文档的处理函数。
    parent (str): 父组名称，默认为根组。
    trans_node (bool): 是否转换节点，None时继承父组设置。
    num_workers (int): 并行处理worker数，0表示串行。
    **kwargs: 其他组参数。
''')

add_english_doc('rag.retriever.TempDocRetriever.add_subretriever', '''
Add a sub-retriever with search configuration.
Args:
    group (str): Target node group name.
    **kwargs: Retriever parameters (e.g., similarity='cosine').
Returns:
    self: For method chaining.
''')

add_chinese_doc('rag.retriever.TempDocRetriever.add_subretriever', '''
添加带搜索配置的子检索器。
Args:
    group (str): 目标节点组名称。
    **kwargs: 检索器参数（如similarity='cosine'）。
Returns:
    self: 支持链式调用。
''')

add_english_doc('rag.doc_node.DocNode', '''
Execute assigned tasks on the specified document.
Args:
    uid (str): Unique identifier.
    content (Union[str, List[Any]]): Node content.
    group (str): Document group name.
    embedding (Dict[str, List[float]]): Dictionary of embedding vectors.
    parent (Union[str, "DocNode"]): Reference to the parent node.
    store: Storage representation.
    node_groups (Dict[str, Dict]): Node storage groups.
    metadata (Dict[str, Any]): Node-level metadata.
    global_metadata (Dict[str, Any]): Document-level metadata.
    text (str): Node content, mutually exclusive with content.
''')

add_chinese_doc('rag.doc_node.DocNode', '''
在指定的文档上执行设定的任务。
Args:
    uid(str): 唯一标识符。
    content(Union[str, List[Any]]):节点内容
    group(str):文档组名
    embedding(Dict[str, List[float]]):嵌入向量字典
    parent(Union[str, "DocNode"]):父节点引用
    store:存储表示
    node_groups(Dict[str, Dict]):节点存储组
    metadata(Dict[str, Any]):节点级元数据
    global_metadata(Dict[str, Any]):文档级元数据
    text(str):节点内容与content互斥
''')

add_english_doc('rag.doc_node.DocNode.get_metadata_str', '''
Get formatted metadata string.
Args:
    mode: MetadataMode.NONE returns an empty string;  
          MetadataMode.LLM filters out metadata not needed by LLM;  
          MetadataMode.EMBED filters out metadata not needed by embedding model;  
          MetadataMode.ALL returns all metadata.
''')

add_chinese_doc('rag.doc_node.DocNode.get_metadata_str', '''
获取格式化元数据字符串
Args:
    mode: MetadataMode.NONE返回空字符串；
          MetadataMode.LLM过滤排除LLM不需要的元数据；
          MetadataMode.EMBED过滤排除嵌入模型不需要的元数据；
          MetadataMode.ALL返回全部元数据。
''')
add_english_doc('rag.doc_node.DocNode.get_text', '''
Combine metadata and content.
Args:
    metadata_mode: Same as the parameter in get_metadata_str.
''')

add_chinese_doc('rag.doc_node.DocNode.get_text', '''
组合元数据和内容
Args:
    metadata_mode: 与get_metadata_str中参数一致
''')
add_english_doc('rag.doc_node.DocNode.has_missing_embedding', '''
Check for missing embedding vectors.
Args:
    embed_keys (Union[str, List[str]]): List of target keys.
''')

add_chinese_doc('rag.doc_node.DocNode.has_missing_embedding', '''
检查缺失的嵌入向量
Args:
    embed_keys(Union[str, List[str]]): 目标键列表
''')
add_english_doc('rag.doc_node.DocNode.do_embedding', '''
Execute embedding computation.
Args:
    embed (Dict[str, Callable]): Target embedding objects.
''')

add_chinese_doc('rag.doc_node.DocNode.do_embedding', '''
执行嵌入计算
Args:
    embed(Dict[str, Callable]): 目标嵌入对象
''')
add_english_doc('rag.doc_node.DocNode.check_embedding_state', '''
Block to check the embedding status and ensure that asynchronous embedding computation is completed.
Args:
    embed_key (str): List of target keys.
''')

add_chinese_doc('rag.doc_node.DocNode.check_embedding_state', '''
阻塞检查嵌入状态,确保异步嵌入计算完成
Args:
    embed_key(str): 目标键列表
''')
add_english_doc('rag.doc_node.DocNode.to_dict', '''
Convert to dictionary format
''')

add_chinese_doc('rag.doc_node.DocNode.to_dict', '''
转换为字典格式
''')
add_english_doc('rag.doc_node.DocNode.with_score', '''
Shallow copy the original node and add a semantic relevance score.
Args:
    score: Relevance score.
''')

add_chinese_doc('rag.doc_node.DocNode.with_score', '''
浅拷贝原节点并添加语义相关分数。
Args:
    score: 相关性得分
''')
add_english_doc('rag.doc_node.DocNode.with_sim_score', '''
Shallow copy the original node and add a similarity score.
Args:
    score: Similarity score.
''')

add_chinese_doc('rag.doc_node.DocNode.with_sim_score', '''
浅拷贝原节点并添加相似度分数。
Args:
    score: 相似度得分
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader', '''
A modular document directory reader that inherits from ModuleBase, supporting reading various document formats from the file system and converting them into standardized DocNode objects.
Args:
    input_dir (Optional[str]): Input directory path. Mutually exclusive with input_files.
    input_files (Optional[List]): Directly specified list of files. Mutually exclusive with input_dir.
    exclude (Optional[List]): List of file patterns to exclude.
    exclude_hidden (bool): Whether to exclude hidden files.
    recursive (bool): Whether to recursively read subdirectories.
    encoding (str): Encoding format of text files.
    required_exts (Optional[List[str]]): Whitelist of file extensions to process.
    file_extractor (Optional[Dict[str, Callable]]): Dictionary of custom file readers.
    fs (Optional[AbstractFileSystem]): Custom file system.
    metadata_genf (Optional[Callable[[str], Dict]]): Metadata generation function that takes a file path and returns a metadata dictionary.
    num_files_limit (Optional[int]): Maximum number of files to read.
    return_trace (bool): Whether to return processing trace information.
    metadatas (Optional[Dict]): Predefined global metadata dictionary.
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader', '''
模块化的文档目录读取器，继承自 ModuleBase，支持从文件系统读取多种格式的文档并转换为标准化的 DocNode 。
Args:
    input_dir (Optional[str]): 输入目录路径。与input_files二选一，不可同时指定。
    input_files (Optional[List]):直接指定的文件列表。与input_dir二选一。
    exclude (Optional[List]):需要排除的文件模式列表。
    exclude_hidden (bool): 是否排除隐藏文件。
    recursive (bool):是否递归读取子目录。
    encoding (str):文本文件的编码格式。
    required_exts (Optional[List[str]]):需要处理的文件扩展名白名单。
    file_extractor (Optional[Dict[str, Callable]]):自定义文件阅读器字典。
    fs (Optional[AbstractFileSystem]):自定义文件系统。
    metadata_genf (Optional[Callable[[str], Dict]]):元数据生成函数，接收文件路径返回元数据字典。
    num_files_limit (Optional[int]):最大读取文件数量限制。
    return_trace (bool):是否返回处理过程追踪信息。
    metadatas (Optional[Dict]):预定义的全局元数据字典。
''')

add_example('rag.dataReader.SimpleDirectoryReader', '''
>>> import lazyllm
>>> from lazyllm.tools.dataReader import SimpleDirectoryReader
>>> reader = SimpleDirectoryReader(input_dir="yourpath/",recursive=True,exclude=["*.tmp"],required_exts=[".pdf", ".docx"])
>>> documents = reader.load_data()
''')


add_english_doc('rag.dataReader.FileReader', '''
File content reader whose main function is to convert various input file formats into concatenated plain text content.
Args:
    input_files (Optional[List]): Directly specified list of input files.
''')

add_chinese_doc('rag.dataReader.FileReader', '''
文件内容读取器，主要功能是将多种格式的输入文件转换为拼接后的纯文本内容。
Args:
    input_files (Optional[List]):直接指定的文件列表。
''')

add_example('rag.dataReader.FileReader', '''
>>> import lazyllm
>>> from lazyllm.tools.dataReader import FileReader
>>> reader = FileReader()
>>> content = reader("yourpath/") 
''')

# ---------------------------------------------------------------------------- #

# rag/transform.py

add_english_doc('SentenceSplitter', '''
Split sentences into chunks of a specified size. You can specify the size of the overlap between adjacent chunks.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
''')

add_chinese_doc('SentenceSplitter', '''
将句子拆分成指定大小的块。可以指定相邻块之间重合部分的大小。

Args:
    chunk_size (int): 拆分之后的块大小
    chunk_overlap (int): 相邻两个块之间重合的内容长度
    num_workers(int):控制并行处理的线程/进程数量
    num_workers(int):控制并行处理的线程/进程数量
''')

add_example('SentenceSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
''')

add_english_doc('LLMParser', '''
A text summarizer and keyword extractor that is responsible for analyzing the text input by the user and providing concise summaries or extracting relevant keywords based on the requested task.

Args:
    llm (TrainableModule): A trainable module.
    language (str): The language type, currently only supports Chinese (zh) and English (en).
    task_type (str): Currently supports two types of tasks: summary and keyword extraction.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
''')

add_chinese_doc('LLMParser', '''
一个文本摘要和关键词提取器，负责分析用户输入的文本，并根据请求任务提供简洁的摘要或提取相关关键词。

Args:
    llm (TrainableModule): 可训练的模块
    language (str): 语言种类，目前只支持中文（zh）和英文（en）
    task_type (str): 目前支持两种任务：摘要（summary）和关键词抽取（keywords）。
    num_workers(int):控制并行处理的线程/进程数量。
    num_workers(int):控制并行处理的线程/进程数量。
''')

add_example('LLMParser', '''
>>> from lazyllm import TrainableModule
>>> from lazyllm.tools.rag import LLMParser
>>> llm = TrainableModule("internlm2-chat-7b")
>>> summary_parser = LLMParser(llm, language="en", task_type="summary")
''')

add_english_doc('LLMParser.transform', '''
Perform the set task on the specified document.

Args:
    node (DocNode): The document on which the extraction task needs to be performed.
''')

add_chinese_doc('LLMParser.transform', '''
在指定的文档上执行设定的任务。

Args:
    node (DocNode): 需要执行抽取任务的文档。
''')

add_example('LLMParser.transform', '''
>>> import lazyllm
>>> from lazyllm.tools import LLMParser
>>> llm = lazyllm.TrainableModule("internlm2-chat-7b").start()
>>> m = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
>>> summary_parser = LLMParser(llm, language="en", task_type="summary")
>>> keywords_parser = LLMParser(llm, language="en", task_type="keywords")
>>> documents = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
>>> rm = lazyllm.Retriever(documents, group_name='CoarseChunk', similarity='bm25', topk=6)
>>> doc_nodes = rm("test")
>>> summary_result = summary_parser.transform(doc_nodes[0])
>>> keywords_result = keywords_parser.transform(doc_nodes[0])
''')

add_english_doc('rag.transform.NodeTransform', '''
Processes document nodes in batch, supporting both single-threaded and multi-threaded modes.
Args:
    num_workers(int): Controls whether multi-threading is enabled (enabled when >0).
''')

add_chinese_doc('rag.transform.NodeTransform', '''
批量处理文档节点，支持单线程/多线程模式。
Args:
    num_workers(int)：控制是否启用多线程（>0 时启用）。
''')

add_example('rag.transform.NodeTransform', '''
>>> import lazyllm
>>> from lazyllm.tools import NodeTransform
>>> node_tran = NodeTransform(num_workers=num_workers)
>>> doc = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
>>> nodes = node_tran.batch_forward(doc, "word_split")
''')

add_english_doc('rag.transform.NodeTransform.batch_forward', '''
Process documents in batch with node group transformation.
Args:
    documents (Union[DocNode, List[DocNode]]): Input node(s) to process.
    node_group (str): Target transformation group name.
    **kwargs: Additional transformation parameters.
''')

add_chinese_doc('rag.transform.NodeTransform.batch_forward', '''
批量处理文档节点并生成指定组的子节点。
Args:
    documents (Union[DocNode, List[DocNode]]): 待处理的输入节点（单个或列表）。
    node_group (str): 目标转换组名称。
    **kwargs: 额外转换参数。
''')

add_english_doc('rag.transform.NodeTransform.transform', '''
[Abstract] Core transformation logic to implement.
Args:
    document (DocNode): Input document node.
    **kwargs: Implementation-specific parameters.
''')

add_chinese_doc('rag.transform.NodeTransform.transform', '''
[抽象方法] 需要子类实现的核心转换逻辑。
Args:
    document (DocNode): 输入文档节点。
    **kwargs: 实现相关的参数。
''')

add_english_doc('rag.transform.NodeTransform.with_name', '''
Set transformer name with optional copying.
Args:
    name (Optional[str]): New name for the transformer.
    copy (bool): Whether to return a copy. Default True.
''')

add_chinese_doc('rag.transform.NodeTransform.with_name', '''
设置转换器名称）。
Args:
    name (Optional[str]): 转换器的新名称。
    copy (bool): 是否返回副本，默认为True。
''')

add_english_doc('rag.transform.TransformArgs', '''
A document transformation parameter container for centralized management of processing configurations.
Args:
    f (Union[str, Callable]): Transformation function or registered function name.Can be either a callable function or a string identifier for registered functions.
    trans_node (bool): Whether to transform node types.When True, modifies the document node structure during processing.
    num_workers (int):Controls parallel processing threads.Values >0.
    kwargs (Dict):Additional parameters passed to the transformation function.
    pattern (Union[str, Callable[[str], bool]]):File name/content matching pattern.
''')

add_chinese_doc('rag.transform.TransformArgs', '''
文档转换参数容器，用于统一管理文档处理中的各类配置参数。
Args:
    f(Union[str, Callable]):转换函数或注册的函数名。
    trans_node(bool):是否转换节点类型。
    num_workers(int)：控制是否启用多线程（>0 时启用）。
    kwargs(Dict):传递给转换函数的额外参数。
    pattern(Union[str, Callable[[str], bool]]):文件名/内容匹配模式。
''')

add_example('rag.transform.TransformArgs', '''
>>> from lazyllm.tools import TransformArgs
>>> args = TransformArgs(f=lambda text: text.lower(),num_workers=4,pattern=r'.*\.md$')
>>>config = {'f': 'parse_pdf','kwargs': {'engine': 'pdfminer'},'trans_node': True}
>>>args = TransformArgs.from_dict(config)
print(args['f'])
print(args.get('unknown'))
''')


add_english_doc('rag.similarity.register_similarity', '''
Similarity computation registration decorator, used for unified registration and management of different types of similarity computation methods.
Args:
    func (Callable): The name of the similarity computation function.
    mode (Literal['text', 'embedding']): 'text' indicates direct text matching, while 'embedding' indicates vector-based similarity computation.
    descend (bool): Controls whether multithreading is enabled (enabled when > 0).
    kwargs (Dict): Whether the results are sorted in descending order of similarity.
    batch (bool): Whether to process nodes in batch.
''')

add_chinese_doc('rag.similarity.register_similarity', '''
相似度计算注册装饰器，用于统一注册和管理不同类型的相似度计算方法。
Args:
    func(Callable):相似度计算函数名。
    mode(Literal['text', 'embedding']):text为文本直接匹配,embedding为向量相似度计算。
    descend(bool)：控制是否启用多线程（>0 时启用）。
    kwargs(Dict):结果是否按相似度降序排列。
    batch(bool):是否批量处理节点。
''')

add_english_doc('rag.transform.NodeTransform', '''
Processes document nodes in batch, supporting both single-threaded and multi-threaded modes.
Args:
    num_workers(int): Controls whether multi-threading is enabled (enabled when >0).
''')

add_chinese_doc('rag.transform.NodeTransform', '''
批量处理文档节点，支持单线程/多线程模式。
Args:
    num_workers(int)：控制是否启用多线程（>0 时启用）。
''')

add_example('rag.transform.NodeTransform', '''
>>> import lazyllm
>>> from lazyllm.tools import NodeTransform
>>> node_tran = NodeTransform(num_workers=num_workers)
>>> doc = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
>>> nodes = node_tran.batch_forward(doc, "word_split")
''')

add_english_doc('rag.transform.NodeTransform.batch_forward', '''
Process documents in batch with node group transformation.
Args:
    documents (Union[DocNode, List[DocNode]]): Input node(s) to process.
    node_group (str): Target transformation group name.
    **kwargs: Additional transformation parameters.
''')

add_chinese_doc('rag.transform.NodeTransform.batch_forward', '''
批量处理文档节点并生成指定组的子节点。
Args:
    documents (Union[DocNode, List[DocNode]]): 待处理的输入节点（单个或列表）。
    node_group (str): 目标转换组名称。
    **kwargs: 额外转换参数。
''')

add_english_doc('rag.transform.NodeTransform.transform', '''
[Abstract] Core transformation logic to implement.
Args:
    document (DocNode): Input document node.
    **kwargs: Implementation-specific parameters.
''')

add_chinese_doc('rag.transform.NodeTransform.transform', '''
[抽象方法] 需要子类实现的核心转换逻辑。
Args:
    document (DocNode): 输入文档节点。
    **kwargs: 实现相关的参数。
''')

add_english_doc('rag.transform.NodeTransform.with_name', '''
Set transformer name with optional copying.
Args:
    name (Optional[str]): New name for the transformer.
    copy (bool): Whether to return a copy. Default True.
''')

add_chinese_doc('rag.transform.NodeTransform.with_name', '''
设置转换器名称）。
Args:
    name (Optional[str]): 转换器的新名称。
    copy (bool): 是否返回副本，默认为True。
''')

add_english_doc('rag.transform.TransformArgs', '''
A document transformation parameter container for centralized management of processing configurations.
Args:
    f (Union[str, Callable]): Transformation function or registered function name.Can be either a callable function or a string identifier for registered functions.
    trans_node (bool): Whether to transform node types.When True, modifies the document node structure during processing.
    num_workers (int):Controls parallel processing threads.Values >0.
    kwargs (Dict):Additional parameters passed to the transformation function.
    pattern (Union[str, Callable[[str], bool]]):File name/content matching pattern.
''')

add_chinese_doc('rag.transform.TransformArgs', '''
文档转换参数容器，用于统一管理文档处理中的各类配置参数。
Args:
    f(Union[str, Callable]):转换函数或注册的函数名。
    trans_node(bool):是否转换节点类型。
    num_workers(int)：控制是否启用多线程（>0 时启用）。
    kwargs(Dict):传递给转换函数的额外参数。
    pattern(Union[str, Callable[[str], bool]]):文件名/内容匹配模式。
''')

add_example('rag.transform.TransformArgs', '''
>>> from lazyllm.tools import TransformArgs
>>> args = TransformArgs(f=lambda text: text.lower(),num_workers=4,pattern=r'.*\.md$')
>>>config = {'f': 'parse_pdf','kwargs': {'engine': 'pdfminer'},'trans_node': True}
>>>args = TransformArgs.from_dict(config)
print(args['f'])
print(args.get('unknown'))
''')


add_english_doc('rag.similarity.register_similarity', '''
Similarity computation registration decorator, used for unified registration and management of different types of similarity computation methods.
Args:
    func (Callable): The name of the similarity computation function.
    mode (Literal['text', 'embedding']): 'text' indicates direct text matching, while 'embedding' indicates vector-based similarity computation.
    descend (bool): Controls whether multithreading is enabled (enabled when > 0).
    kwargs (Dict): Whether the results are sorted in descending order of similarity.
    batch (bool): Whether to process nodes in batch.
''')

add_chinese_doc('rag.similarity.register_similarity', '''
相似度计算注册装饰器，用于统一注册和管理不同类型的相似度计算方法。
Args:
    func(Callable):相似度计算函数名。
    mode(Literal['text', 'embedding']):text为文本直接匹配,embedding为向量相似度计算。
    descend(bool)：控制是否启用多线程（>0 时启用）。
    kwargs(Dict):结果是否按相似度降序排列。
    batch(bool):是否批量处理节点。
''')

# ---------------------------------------------------------------------------- #

# rag/doc_manager.py

add_chinese_doc('rag.DocManager', """
DocManager类管理文档列表及相关操作，并通过API提供文档上传、删除、分组等功能。

Args:
    dlm (DocListManager): 文档列表管理器，用于处理具体的文档操作。

""")

add_chinese_doc('rag.DocManager.document', """
提供默认文档页面的重定向接口。

**Returns:**\n
- RedirectResponse: 重定向到 `/docs` 页面。
""")

add_chinese_doc('rag.DocManager.list_kb_groups', """
列出所有文档分组的接口。

**Returns:**\n
- BaseResponse: 包含所有文档分组的数据。
""")

add_chinese_doc('rag.DocManager.upload_files', """
上传文件并更新其状态的接口。可以同时上传多个文件。

Args:
    files (List[UploadFile]): 上传的文件列表。
    override (bool): 是否覆盖已存在的文件。默认为False。
    metadatas (Optional[str]): 文件的元数据，JSON格式。
    user_path (Optional[str]): 用户自定义的文件上传路径。

**Returns:**\n
- BaseResponse: 上传结果和文件ID。
""")

add_chinese_doc('rag.doc_manager.DocManager.add_files', """
批量添加文件。
Args:
    files (List[UploadFile]): 上传的文件列表。
    group_name (str): 目标知识库分组名称，为空时不添加到分组。
    metadatas (Optional[str]): 文件的元数据，JSON格式。
**Returns:**\n
- BaseResponse:返回所有输入文件对应的唯一文件ID列表，包括新增和已存在的文件。若出现异常，则返回错误码和异常信息。
""")

add_chinese_doc('rag.doc_manager.DocManager.add_files', """
批量添加文件。
Args:
    files (List[UploadFile]): 上传的文件列表。
    group_name (str): 目标知识库分组名称，为空时不添加到分组。
    metadatas (Optional[str]): 文件的元数据，JSON格式。
**Returns:**\n
- BaseResponse:返回所有输入文件对应的唯一文件ID列表，包括新增和已存在的文件。若出现异常，则返回错误码和异常信息。
""")

add_chinese_doc('rag.DocManager.list_files', """
列出已上传文件的接口。

Args:
    limit (Optional[int]): 返回的文件数量限制。默认为None。
    details (bool): 是否返回详细信息。默认为True。
    alive (Optional[bool]): 如果为True，只返回未删除的文件。默认为None。

**Returns:**\n
- BaseResponse: 文件列表数据。
""")

add_chinese_doc('rag.DocManager.list_files_in_group', """
列出指定分组中文件的接口。

Args:
    group_name (Optional[str]): 文件分组名称。
    limit (Optional[int]): 返回的文件数量限制。默认为None。
    alive (Optional[bool]): 是否只返回未删除的文件。

**Returns:**\n
- BaseResponse: 分组文件列表。
""")

add_chinese_doc('rag.DocManager.add_files_to_group_by_id', """
通过文件ID将文件添加到指定分组的接口。

Args:
    request (FileGroupRequest): 包含文件ID和分组名称的请求。

**Returns:**\n
- BaseResponse: 操作结果。
""")

add_chinese_doc('rag.DocManager.add_files_to_group', """
将文件上传后直接添加到指定分组的接口。

Args:
    files (List[UploadFile]): 上传的文件列表。
    group_name (str): 要添加到的分组名称。
    override (bool): 是否覆盖已存在的文件。默认为False。
    metadatas (Optional[str]): 文件元数据，JSON格式。
    user_path (Optional[str]): 用户自定义的文件上传路径。

**Returns:**\n
- BaseResponse: 操作结果和文件ID。
""")

add_chinese_doc('rag.DocManager.delete_files', """
删除指定文件的接口。

Args:
    request (FileGroupRequest): 包含文件ID和分组名称的请求。

**Returns:**\n
- BaseResponse: 删除操作结果。
""")

add_english_doc('rag.DocManager', """
The `DocManager` class manages document lists and related operations, providing APIs for uploading, deleting, and grouping documents.

Args:
    dlm (DocListManager): Document list manager responsible for handling document-related operations.

""")

add_english_doc('rag.DocManager.document', """
An endpoint to redirect to the default documentation page.

**Returns:**\n
- RedirectResponse: Redirects to the `/docs` page.
""")

add_english_doc('rag.DocManager.list_kb_groups', """
An endpoint to list all document groups.

**Returns:**\n
- BaseResponse: Contains the data of all document groups.
""")

add_english_doc('rag.DocManager.upload_files', """
An endpoint to upload files and update their status. Multiple files can be uploaded at once.

Args:
    files (List[UploadFile]): List of files to upload.
    override (bool): Whether to overwrite existing files. Default is False.
    metadatas (Optional[str]): Metadata for the files in JSON format.
    user_path (Optional[str]): User-defined path for file uploads.

**Returns:**\n
- BaseResponse: Upload results and file IDs.
""")

add_english_doc('rag.DocManager.add_files', """
Batch add files.
Args:
    files (List[UploadFile]): List of uploaded files.
    group_name (str): Target knowledge base group name; if empty, files are not added to any group.
    metadatas (Optional[str]): Metadata of the files in JSON format.
**Returns:**\n
- BaseResponse: Returns a list of unique file IDs corresponding to all input files, including newly added and existing ones. In case of exceptions, returns error codes and exception information.
""")

add_english_doc('rag.DocManager.add_files', """
Batch add files.
Args:
    files (List[UploadFile]): List of uploaded files.
    group_name (str): Target knowledge base group name; if empty, files are not added to any group.
    metadatas (Optional[str]): Metadata of the files in JSON format.
**Returns:**\n
- BaseResponse: Returns a list of unique file IDs corresponding to all input files, including newly added and existing ones. In case of exceptions, returns error codes and exception information.
""")

add_english_doc('rag.DocManager.list_files', """
An endpoint to list uploaded files.

Args:
    limit (Optional[int]): Limit on the number of files returned. Default is None.
    details (bool): Whether to return detailed information. Default is True.
    alive (Optional[bool]): If True, only returns non-deleted files. Default is None.

**Returns:**\n
- BaseResponse: File list data.
""")

add_english_doc('rag.DocManager.list_files_in_group', """
An endpoint to list files in a specific group.

Args:
    group_name (Optional[str]): The name of the file group.
    limit (Optional[int]): Limit on the number of files returned. Default is None.
    alive (Optional[bool]): Whether to return only non-deleted files.

**Returns:**\n
- BaseResponse: List of files in the group.
""")

add_english_doc('rag.DocManager.add_files_to_group_by_id', """
An endpoint to add files to a specific group by file IDs.

Args:
    request (FileGroupRequest): Request containing file IDs and group name.

**Returns:**\n
- BaseResponse: Operation result.
""")

add_english_doc('rag.DocManager.add_files_to_group', """
An endpoint to upload files and directly add them to a specified group.

Args:
    files (List[UploadFile]): List of files to upload.
    group_name (str): Name of the group to add the files to.
    override (bool): Whether to overwrite existing files. Default is False.
    metadatas (Optional[str]): Metadata for the files in JSON format.
    user_path (Optional[str]): User-defined path for file uploads.

**Returns:**\n
- BaseResponse: Operation result and file IDs.
""")

add_english_doc('rag.DocManager.delete_files', """
An endpoint to delete specified files.

Args:
    request (FileGroupRequest): Request containing file IDs and group name.

**Returns:**\n
- BaseResponse: Deletion operation result.
""")

# ---------------------------------------------------------------------------- #

# rag/utils.py

add_chinese_doc('rag.DocListManager.table_inited', """\
检查数据库表是否已初始化。

**Returns:**
- bool: 如果表已初始化，则返回True；否则返回False。
""")

add_chinese_doc('rag.DocListManager._init_tables', """\
初始化数据库表。此方法应在未初始化表时调用，用于创建必要的表结构。
""")

add_chinese_doc('rag.DocListManager.list_files', """\
列出符合条件的文件。

Args:
    limit (int, optional): 要返回的文件数限制。
    details (bool): 如果为True，则返回文件的详细信息。
    status (str or list of str, optional): 要筛选的文件状态。
    exclude_status (str or list of str, optional): 要排除的文件状态。

**Returns:**
- list: 文件列表。
""")

add_chinese_doc('rag.DocListManager.list_all_kb_group', """\
列出所有知识库分组的名称。

**Returns:**
- list: 知识库分组名称列表。
""")

add_chinese_doc('rag.DocListManager.add_kb_group', """\
添加一个新的知识库分组。

Args:
    name (str): 要添加的分组名称。
""")

add_chinese_doc('rag.DocListManager.list_kb_group_files', """\
列出指定知识库分组中的文件。

Args:
    group (str, optional): 分组名称。默认为None，表示所有分组。
    limit (int, optional): 要返回的文件数限制。
    details (bool): 如果为True，则返回文件的详细信息。
    status (str or list of str, optional): 要筛选的文件状态。
    exclude_status (str or list of str, optional): 要排除的文件状态。
    upload_status (str, optional): 要筛选的上传状态。
    exclude_upload_status (str or list of str, optional): 要排除的上传状态。

**Returns:**
- list: 文件列表。
""")

add_chinese_doc('rag.DocListManager.add_files', """\
将文件添加到数据库中。

Args:
    files (list of str): 要添加的文件路径列表。
    metadatas (list, optional): 与文件相关的元数据。
    status (str, optional): 文件状态。

**Returns:**
- list: 文件的ID列表。
""")

add_chinese_doc('rag.DocListManager.update_file_message', """\
更新指定文件的消息。

Args:
    fileid (str): 文件ID。
    **kw: 需要更新的其他键值对。
""")

add_chinese_doc('rag.DocListManager.add_files_to_kb_group', """\
将文件添加到指定的知识库分组中。

Args:
    file_ids (list of str): 要添加的文件ID列表。
    group (str): 要添加的分组名称。
""")

add_chinese_doc('rag.DocListManager.delete_files', """\
将与文件关联的知识库条目设为删除中，并由各知识库进行异步删除解析结果及关联记录。

Args:
    file_ids (list of str): 要删除的文件ID列表
""")

add_chinese_doc('rag.DocListManager.delete_files_from_kb_group', """\
从指定的知识库分组中删除文件。

Args:
    file_ids (list of str): 要删除的文件ID列表。
    group (str): 分组名称。
""")

add_chinese_doc('rag.DocListManager.get_file_status', """\
获取指定文件的状态。

Args:
    fileid (str): 文件ID。

**Returns:**
- str: 文件的当前状态。
""")

add_chinese_doc('rag.DocListManager.update_file_status', """\
更新指定文件的状态。

Args:
    file_ids (list of str): 要更新状态的文件ID列表。
    status (str): 新的文件状态。
""")

add_chinese_doc('rag.DocListManager.update_kb_group', """\
更新指定知识库分组中的内容。

Args:
    cond_file_ids (list of str, optional): 过滤使用的文件ID列表，默认为None。
    cond_group (str, optional): 过滤使用的知识库分组名称，默认为None。
    cond_status_list (list of str, optional): 过滤使用的状态列表，默认为None。
    new_status (str, optional): 新状态, 默认为None。
    new_need_reparse (bool, optinoal): 新的是否需重解析标志。

**Returns:**
- list: 得到更新的列表list of (doc_id, group_name)
""")

add_chinese_doc('rag.DocListManager.release', """\
释放当前管理器的资源。

""")

add_english_doc('rag.DocListManager.table_inited', """\
Checks if the database tables have been initialized.

**Returns:**
- bool: True if the tables have been initialized, False otherwise.
""")

add_english_doc('rag.DocListManager._init_tables', """\
Initializes the database tables. This method should be called when the tables have not been initialized yet, creating the necessary table structures.
""")

add_english_doc('rag.DocListManager.list_files', """\
Lists files that meet the specified criteria.

Args:
    limit (int, optional): Limit on the number of files to return.
    details (bool): If True, return detailed file information.
    status (str or list of str, optional): Filter files by status.
    exclude_status (str or list of str, optional): Exclude files with these statuses.

**Returns:**
- list: List of files.
""")

add_english_doc('rag.DocListManager.list_all_kb_group', """\
Lists all the knowledge base group names.

**Returns:**
- list: List of knowledge base group names.
""")

add_english_doc('rag.DocListManager.add_kb_group', """\
Adds a new knowledge base group.

Args:
    name (str): Name of the group to add.
""")

add_english_doc('rag.DocListManager.list_kb_group_files', """\
Lists files in the specified knowledge base group.

Args:
    group (str, optional): Group name. Defaults to None, meaning all groups.
    limit (int, optional): Limit on the number of files to return.
    details (bool): If True, return detailed file information.
    status (str or list of str, optional): Filter files by status.
    exclude_status (str or list of str, optional): Exclude files with these statuses.
    upload_status (str, optional): Filter by upload status.
    exclude_upload_status (str or list of str, optional): Exclude files with these upload statuses.

**Returns:**
- list: List of files.
""")

add_english_doc('rag.DocListManager.add_files', """\
Adds files to the database.

Args:
    files (list of str): List of file paths to add.
    metadatas (list, optional): Metadata associated with the files.
    status (str, optional): File status.

**Returns:**
- list: List of file IDs.
""")

add_english_doc('rag.DocListManager.update_file_message', """\
Updates the message for a specified file.

Args:
    fileid (str): File ID.
    **kw: Additional key-value pairs to update.
""")

add_english_doc('rag.DocListManager.add_files_to_kb_group', """\
Adds files to the specified knowledge base group.

Args:
    file_ids (list of str): List of file IDs to add.
    group (str): Name of the group to add the files to.
""")

add_english_doc('rag.DocListManager.delete_files', """\
Set the knowledge base entries associated with the document to "deleting," and have each knowledge base asynchronously delete parsed results and associated records.

Args:
    file_ids (list of str): List of file IDs to delete.
""")

add_english_doc('rag.DocListManager.delete_files_from_kb_group', """\
Deletes files from the specified knowledge base group.

Args:
    file_ids (list of str): List of file IDs to delete.
    group (str): Name of the group.
""")

add_english_doc('rag.DocListManager.get_file_status', """\
Retrieves the status of a specified file.

Args:
    fileid (str): File ID.

**Returns:**
- str: The current status of the file.
""")

add_english_doc('rag.DocListManager.update_kb_group', """\
Updates the record of kb_group_document.

Args:
    cond_file_ids (list of str, optional): a list of file IDs to filter by, default None.
    cond_group (str, optional): a kb_group name to filter by, default None.
    cond_status_list (list of str, optional): a list of statuses to filter by, default None.
    new_status (str, optional): the new status to update to, default None
    new_need_reparse (bool, optinoal): the new need_reparse flag to update to, default None

**Returns:**
- list: updated records, list of (doc_id, group_name)
""")

add_english_doc('rag.DocListManager.release', """\
Releases the resources of the current manager.
""")

add_chinese_doc('IndexBase.update', '''\
更新索引内容。

该方法接收一组文档节点对象，并将其添加或更新到索引结构中。通常用于增量构建或刷新索引。

Args:
    nodes (List[DocNode]): 需要更新的文档节点列表。
''')

add_english_doc('IndexBase.update', '''\
Update index contents.

This method receives a list of document nodes and updates or inserts them into the index structure. Typically used for incremental indexing or refreshing data.

Args:
    nodes (List[DocNode]): A list of document nodes to update or insert.
''')

add_chinese_doc('IndexBase.remove', '''\
从索引中移除指定文档节点。

可根据唯一标识符列表删除索引中的文档节点，可选地指定组名称以限定范围。

Args:
    uids (List[str]): 需要移除的文档节点的唯一标识符列表。
    group_name (Optional[str]): 可选的组名称，用于限定要删除的范围。
''')

add_english_doc('IndexBase.remove', '''\
Remove specific document nodes from the index.

Removes document nodes based on their unique identifiers, optionally scoped by group name.

Args:
    uids (List[str]): List of unique IDs corresponding to the document nodes to remove.
    group_name (Optional[str]): Optional group name to scope the removal operation.
''')

add_chinese_doc('IndexBase.query', '''\
执行索引查询。

根据传入的参数执行查询操作，返回匹配的文档节点列表。具体查询逻辑由实现类定义。

Returns:
    List[DocNode]: 查询结果的文档节点列表。
''')

add_english_doc('IndexBase.query', '''\
Execute a query over the index.

Performs a query based on the given arguments and returns matching document nodes. The logic depends on the specific implementation.

Returns:
    List[DocNode]: A list of matched document nodes from the index.
''')

# ---------------------------------------------------------------------------- #

add_chinese_doc('WebModule', '''\
WebModule是LazyLLM为开发者提供的基于Web的交互界面。在初始化并启动一个WebModule之后，开发者可以从页面上看到WebModule背后的模块结构，并将Chatbot组件的输入传输给自己开发的模块进行处理。
模块返回的结果和日志会直接显示在网页的“处理日志”和Chatbot组件上。除此之外，WebModule支持在网页上动态加入Checkbox或Text组件用于向模块发送额外的参数。
WebModule页面还提供“使用上下文”，“流式输出”和“追加输出”的Checkbox，可以用来改变页面和后台模块的交互方式。

<span style="font-size: 20px;">&ensp;**`WebModule.init_web(component_descs) -> gradio.Blocks`**</span>
使用gradio库生成演示web页面，初始化session相关数据以便在不同的页面保存各自的对话和日志，然后使用传入的component_descs参数为页面动态添加Checkbox和Text组件，最后设置页面上的按钮和文本框的相应函数
之后返回整个页面。WebModule的__init__函数调用此方法生成页面。

Args:
    component_descs (list): 用于动态向页面添加组件的列表。列表中的每个元素也是一个列表，其中包含5个元素，分别是组件对应的模块ID，模块名，组件名，组件类型（目前仅支持Checkbox和Text），组件默认值。
    m (Any): 模型或模块实例，通常为 FlowBase、ActionModule 或 ChatModule。
    components (Dict[Any, Any]): 组件配置映射，支持模块及其功能定义。
    title (str): Web 页面标题，默认为 "对话演示终端"。
    port (Union[int, range, tuple, list], optional): 指定 Web 服务的端口范围。
    history (List[Any]): 历史模块 ID 列表，用于记录上下文。
    text_mode (Optional[Mode]): 文本更新模式，支持 Dynamic、Refresh、Appendix。
    trace_mode (Optional[Mode]): 已废弃的参数，不推荐使用。
    audio (bool): 是否启用语音输入组件。
    stream (bool): 是否启用模型输出流式展示。
    files_target (Union[Any, List[Any]], optional): 接收文件上传的目标模块。
    static_paths (Union[str, Path, List[Union[str, Path]]], optional): 本地静态资源路径。
    encode_files (bool): 是否对上传文件进行 Base64 编码。
    share (bool): 是否启用 Gradio 的 public share 功能（需联网）。
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
    m (Any): A model or module instance, typically FlowBase, ActionModule, or ChatModule.
    components (Dict[Any, Any]): Component bindings, mapping modules to tool functions.
    title (str): Title of the web interface page (default: "对话演示终端").
    port (Union[int, range, tuple, list], optional): Port or range of ports to serve the web UI.
    history (List[Any]): Optional list of modules to inject into chat history.
    text_mode (Optional[Mode]): Text update mode: Dynamic, Refresh, or Appendix.
    trace_mode (Optional[Mode]): Deprecated.
    audio (bool): Enable audio input components.
    stream (bool): Enable streaming output from the model.
    files_target (Union[Any, List[Any]], optional): Target modules for uploaded files.
    static_paths (Union[str, Path, List[Union[str, Path]]], optional): Local static file paths to expose.
    encode_files (bool): Whether to base64-encode uploaded files.
    share (bool): Enable Gradio public sharing (requires internet).
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

add_chinese_doc('WebModule.init_web', '''\
初始化 Web UI 页面。

该方法使用 Gradio 构建对话界面，并将组件绑定到事件，支持会话选择、流式输出、上下文控制、多模态输入等功能。该方法返回构建完成的 Gradio Blocks 对象。

Args:
    component_descs (List[Tuple]): 组件描述列表，每项为五元组 (module, group_name, name, component_type, value)，
        例如：('MyModule', 'GroupA', 'use_cache', 'Checkbox', True)。

Returns:
    gr.Blocks: 构建好的 Gradio 页面对象，可用于 launch 启动 Web 服务。
''')

add_english_doc('WebModule.init_web', '''\
Initialize the Web UI page.

This method uses Gradio to build the interactive chat interface and binds all components to the appropriate logic. It supports session selection, streaming output, context toggling, multimodal input, and control tools. The method returns the constructed Gradio Blocks object.

Args:
    component_descs (List[Tuple]): A list of component descriptors. Each element is a 5-tuple 
        (module, group_name, name, component_type, value), e.g. ('MyModule', 'GroupA', 'use_cache', 'Checkbox', True).

Returns:
    gr.Blocks: The constructed Gradio UI object, which can be launched via `.launch()`.
''')

add_chinese_doc('WebModule.wait', '''\
阻塞主线程，等待 Web 页面关闭。

该方法会阻塞当前线程直到 Web 页面（Gradio demo）被关闭，适用于部署后阻止程序提前退出的场景。
''')

add_english_doc('WebModule.wait', '''\
Block the main thread until the web interface is closed.

This method blocks the current thread until the Gradio demo is closed. Useful in deployment scenarios to prevent premature program exit.
''')

add_chinese_doc('WebModule.stop', '''\
关闭 Web 页面并清理资源。

如果 Web 页面已初始化，则关闭 Gradio demo，释放资源并重置 `demo` 与 `url` 属性。
''')

add_english_doc('WebModule.stop', '''\
Stop the web interface and clean up resources.

If the web demo has been initialized, this method closes the Gradio demo, frees related resources, and resets `demo` and `url` attributes.
''')

add_chinese_doc('CodeGenerator', '''\
代码生成模块。

该模块基于用户提供的提示词生成代码，会根据提示内容自动选择中文或英文的系统提示词，并从输出中提取 Python 代码片段。

`__init__(self, base_model, prompt="")`
初始化代码生成器。

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): 模型路径字符串，或已初始化的模型实例。
    prompt (str): 用户自定义的代码生成提示词，可为中文或英文。
''')


add_english_doc('CodeGenerator', '''\
Code Generation Module.

This module generates code based on a user-defined prompt. It automatically selects a Chinese or English system prompt based on the input, and extracts Python code snippets from the output.

`__init__(self, base_model, prompt="")`
Initializes the code generator with a base model and prompt.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A path string to load the model, or an initialized model instance.
    prompt (str): A user-defined prompt to guide the code generation. May contain Chinese or English.
''')

add_example('CodeGenerator', ['''\
>>> from lazyllm.components import CodeGenerator
>>> generator = CodeGenerator(base_model="deepseek-coder", prompt="写一个Python函数，计算斐波那契数列。")
>>> result = generator("请给出实现代码")
>>> print(result)
... def fibonacci(n):
...     if n <= 1:
...         return n
...     return fibonacci(n-1) + fibonacci(n-2)
'''])

#actors/parameter_extractor
add_chinese_doc('ParameterExtractor', '''\
参数提取模块。

该模块根据参数名称、类型、描述和是否必填，从文本中提取结构化参数，底层依赖语言模型实现。

`__init__(self, base_model, param, type, description, require)`
使用参数定义和模型初始化参数提取器。

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): 用于参数提取的模型路径或模型实例。
    param (list[str]): 需要提取的参数名称列表。
    type (list[str]): 参数类型列表，如 "int"、"str"、"bool" 等。
    description (list[str]): 每个参数的描述信息。
    require (list[bool]): 每个参数是否为必填项的布尔列表。
''')

add_english_doc('ParameterExtractor', '''\
Parameter Extraction Module.

This module extracts structured parameters from a given text using a language model, based on the parameter names, types, descriptions, and whether they are required.

`__init__(self, base_model, param, type, description, require)`
Initializes the parameter extractor with the parameter specification and base model.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A model path or model instance used for extraction.
    param (list[str]): List of parameter names to extract.
    type (list[str]): List of parameter types (e.g., "int", "str", "bool").
    description (list[str]): List of descriptions for each parameter.
    require (list[bool]): List indicating whether each parameter is required.
''')

add_example('ParameterExtractor', ['''\
>>> from lazyllm.components import ParameterExtractor
>>> extractor = ParameterExtractor(
...     base_model="deepseek-chat",
...     param=["name", "age"],
...     type=["str", "int"],
...     description=["The user's name", "The user's age"],
...     require=[True, True]
... )
>>> result = extractor("My name is Alice and I am 25 years old.")
>>> print(result)
... ['Alice', 25]
'''])

# actors/question_rewrite.py
add_chinese_doc('QustionRewrite', '''\
问题改写模块。

该模块使用语言模型对用户输入的问题进行改写，可根据输出格式选择返回字符串或列表。

`__init__(self, base_model, rewrite_prompt="", formatter="str")`
使用提示词和模型初始化问题改写模块。

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): 问题改写所使用的模型路径或已初始化模型。
    rewrite_prompt (str): 用户自定义的改写提示词。
    formatter (str): 输出格式，可选 "str"（字符串）或 "list"（按行分割的列表）。
''')

add_english_doc('QustionRewrite', '''\
Question Rewrite Module.

This module rewrites or reformulates a user query using a language model. It supports both string and list output formats based on the formatter.

`__init__(self, base_model, rewrite_prompt="", formatter="str")`
Initializes the question rewrite module with a prompt and model.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A path string or initialized model for question rewriting.
    rewrite_prompt (str): Custom prompt to guide the rewrite behavior.
    formatter (str): Output format type; either "str" or "list".
''')

add_example('QustionRewrite', ['''\
>>> from lazyllm.components import QustionRewrite
>>> rewriter = QustionRewrite(base_model="chatglm", rewrite_prompt="请将问题改写为更适合检索的形式", formatter="list")
>>> result = rewriter("中国的最高山峰是什么？")
>>> print(result)
... ['中国的最高山峰是哪一座？', '中国海拔最高的山是什么？']
'''])


add_chinese_doc('WebModule.init_web', '''\
初始化 Web UI 页面。

该方法使用 Gradio 构建对话界面，并将组件绑定到事件，支持会话选择、流式输出、上下文控制、多模态输入等功能。该方法返回构建完成的 Gradio Blocks 对象。

Args:
    component_descs (List[Tuple]): 组件描述列表，每项为五元组 (module, group_name, name, component_type, value)，
        例如：('MyModule', 'GroupA', 'use_cache', 'Checkbox', True)。

Returns:
    gr.Blocks: 构建好的 Gradio 页面对象，可用于 launch 启动 Web 服务。
''')

add_english_doc('WebModule.init_web', '''\
Initialize the Web UI page.

This method uses Gradio to build the interactive chat interface and binds all components to the appropriate logic. It supports session selection, streaming output, context toggling, multimodal input, and control tools. The method returns the constructed Gradio Blocks object.

Args:
    component_descs (List[Tuple]): A list of component descriptors. Each element is a 5-tuple 
        (module, group_name, name, component_type, value), e.g. ('MyModule', 'GroupA', 'use_cache', 'Checkbox', True).

Returns:
    gr.Blocks: The constructed Gradio UI object, which can be launched via `.launch()`.
''')

add_chinese_doc('WebModule.wait', '''\
阻塞主线程，等待 Web 页面关闭。

该方法会阻塞当前线程直到 Web 页面（Gradio demo）被关闭，适用于部署后阻止程序提前退出的场景。
''')

add_english_doc('WebModule.wait', '''\
Block the main thread until the web interface is closed.

This method blocks the current thread until the Gradio demo is closed. Useful in deployment scenarios to prevent premature program exit.
''')

add_chinese_doc('WebModule.stop', '''\
关闭 Web 页面并清理资源。

如果 Web 页面已初始化，则关闭 Gradio demo，释放资源并重置 `demo` 与 `url` 属性。
''')

add_english_doc('WebModule.stop', '''\
Stop the web interface and clean up resources.

If the web demo has been initialized, this method closes the Gradio demo, frees related resources, and resets `demo` and `url` attributes.
''')

#actors/codegenerator
add_chinese_doc('CodeGenerator', '''\
代码生成模块。

该模块基于用户提供的提示词生成代码，会根据提示内容自动选择中文或英文的系统提示词，并从输出中提取 Python 代码片段。

`__init__(self, base_model, prompt="")`
初始化代码生成器。

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): 模型路径字符串，或已初始化的模型实例。
    prompt (str): 用户自定义的代码生成提示词，可为中文或英文。
''')


add_english_doc('CodeGenerator', '''\
Code Generation Module.

This module generates code based on a user-defined prompt. It automatically selects a Chinese or English system prompt based on the input, and extracts Python code snippets from the output.

`__init__(self, base_model, prompt="")`
Initializes the code generator with a base model and prompt.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A path string to load the model, or an initialized model instance.
    prompt (str): A user-defined prompt to guide the code generation. May contain Chinese or English.
''')

add_example('CodeGenerator', ['''\
>>> from lazyllm.components import CodeGenerator
>>> generator = CodeGenerator(base_model="deepseek-coder", prompt="写一个Python函数，计算斐波那契数列。")
>>> result = generator("请给出实现代码")
>>> print(result)
... def fibonacci(n):
...     if n <= 1:
...         return n
...     return fibonacci(n-1) + fibonacci(n-2)
'''])

add_chinese_doc('ParameterExtractor', '''\
参数提取模块。

该模块根据参数名称、类型、描述和是否必填，从文本中提取结构化参数，底层依赖语言模型实现。

`__init__(self, base_model, param, type, description, require)`
使用参数定义和模型初始化参数提取器。

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): 用于参数提取的模型路径或模型实例。
    param (list[str]): 需要提取的参数名称列表。
    type (list[str]): 参数类型列表，如 "int"、"str"、"bool" 等。
    description (list[str]): 每个参数的描述信息。
    require (list[bool]): 每个参数是否为必填项的布尔列表。
''')

add_english_doc('ParameterExtractor', '''\
Parameter Extraction Module.

This module extracts structured parameters from a given text using a language model, based on the parameter names, types, descriptions, and whether they are required.

`__init__(self, base_model, param, type, description, require)`
Initializes the parameter extractor with the parameter specification and base model.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A model path or model instance used for extraction.
    param (list[str]): List of parameter names to extract.
    type (list[str]): List of parameter types (e.g., "int", "str", "bool").
    description (list[str]): List of descriptions for each parameter.
    require (list[bool]): List indicating whether each parameter is required.
''')

add_example('ParameterExtractor', ['''\
>>> from lazyllm.components import ParameterExtractor
>>> extractor = ParameterExtractor(
...     base_model="deepseek-chat",
...     param=["name", "age"],
...     type=["str", "int"],
...     description=["The user's name", "The user's age"],
...     require=[True, True]
... )
>>> result = extractor("My name is Alice and I am 25 years old.")
>>> print(result)
... ['Alice', 25]
'''])

add_chinese_doc('QustionRewrite', '''\
问题改写模块。

该模块使用语言模型对用户输入的问题进行改写，可根据输出格式选择返回字符串或列表。

`__init__(self, base_model, rewrite_prompt="", formatter="str")`
使用提示词和模型初始化问题改写模块。

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): 问题改写所使用的模型路径或已初始化模型。
    rewrite_prompt (str): 用户自定义的改写提示词。
    formatter (str): 输出格式，可选 "str"（字符串）或 "list"（按行分割的列表）。
''')

add_english_doc('QustionRewrite', '''\
Question Rewrite Module.

This module rewrites or reformulates a user query using a language model. It supports both string and list output formats based on the formatter.

`__init__(self, base_model, rewrite_prompt="", formatter="str")`
Initializes the question rewrite module with a prompt and model.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A path string or initialized model for question rewriting.
    rewrite_prompt (str): Custom prompt to guide the rewrite behavior.
    formatter (str): Output format type; either "str" or "list".
''')

add_example('QustionRewrite', ['''\
>>> from lazyllm.components import QustionRewrite
>>> rewriter = QustionRewrite(base_model="chatglm", rewrite_prompt="请将问题改写为更适合检索的形式", formatter="list")
>>> result = rewriter("中国的最高山峰是什么？")
>>> print(result)
... ['中国的最高山峰是哪一座？', '中国海拔最高的山是什么？']
'''])


add_chinese_doc('ToolManager', '''\
ToolManager是一个工具管理类，用于提供工具信息和工具调用给function call。

此管理类构造时需要传入工具名字符串列表。此处工具名可以是LazyLLM提供的，也可以是用户自定义的，如果是用户自定义的，首先需要注册进LazyLLM中才可以使用。在注册时直接使用 `fc_register` 注册器，该注册器已经建立 `tool` group，所以使用该工具管理类时，所有函数都统一注册进 `tool` 分组即可。待注册的函数需要对函数参数进行注解，并且需要对函数增加功能描述，以及参数类型和作用描述。以方便工具管理类能对函数解析传给LLM使用。

Args:
    tools (List[str]): 工具名称字符串列表。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
''')

add_english_doc('ToolManager', '''\
ToolManager is a tool management class used to provide tool information and tool calls to function call.

When constructing this management class, you need to pass in a list of tool name strings. The tool name here can be provided by LazyLLM or user-defined. If it is user-defined, it must first be registered in LazyLLM before it can be used. When registering, directly use the `fc_register` registrar, which has established the `tool` group, so when using the tool management class, all functions can be uniformly registered in the `tool` group. The function to be registered needs to annotate the function parameters, and add a functional description to the function, as well as the parameter type and function description. This is to facilitate the tool management class to parse the function and pass it to LLM for use.

Args:
    tools (List[str]): A list of tool name strings.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
''')

add_example('ToolManager', """\
>>> from lazyllm.tools import ToolManager, fc_register
>>> import json
>>> from typing import Literal
>>> @fc_register("tool")
>>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
...     '''
...     Get the current weather in a given location
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         unit (str): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'fahrenheit'})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> @fc_register("tool")
>>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
...     '''
...     Get an N-day weather forecast
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         num_days (int): The number of days to forecast.
...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> tools = ["get_current_weather", "get_n_day_weather_forecast"]
>>> tm = ToolManager(tools)
>>> print(tm([{'name': 'get_n_day_weather_forecast', 'arguments': {'location': 'Beijing', 'num_days': 3}}])[0])
'{"location": "Beijing", "temperature": "85", "unit": "fahrenheit", "num_days": 3}'
""")

add_chinese_doc('ModuleTool', '''\
用于构建工具模块的基类。

该类封装了函数签名和文档字符串的自动解析逻辑，可生成标准化的参数模式（基于 pydantic），并对输入进行校验和工具调用的标准封装。

`__init__(self, verbose=False, return_trace=True)`
初始化工具模块。

Args:
    verbose (bool): 是否在执行过程中输出详细日志。
    return_trace (bool): 是否在结果中保留中间执行痕迹。
''')

add_english_doc('ModuleTool', '''\
Base class for defining tools using callable Python functions.

This class automatically parses function signatures and docstrings to build a parameter schema using `pydantic`. It also performs input validation and handles standardized tool execution.

`__init__(self, verbose=False, return_trace=True)`
Initializes a tool wrapper module.

Args:
    verbose (bool): Whether to print verbose logs during execution.
    return_trace (bool): Whether to keep intermediate execution trace in the result.
''')

add_example('ModuleTool', """
>>> from lazyllm.components import ModuleTool
>>> class AddTool(ModuleTool):
...     def apply(self, a: int, b: int) -> int:
...         '''Add two integers.
...         
...         Args:
...             a (int): First number.
...             b (int): Second number.
...         
...         Returns:
...             int: The sum of a and b.
...         '''
...         return a + b
>>> tool = AddTool()
>>> result = tool({'a': 3, 'b': 5})
>>> print(result)
8
""")


add_chinese_doc('ModuleTool', '''\
用于构建工具模块的基类。

该类封装了函数签名和文档字符串的自动解析逻辑，可生成标准化的参数模式（基于 pydantic），并对输入进行校验和工具调用的标准封装。

`__init__(self, verbose=False, return_trace=True)`
初始化工具模块。

Args:
    verbose (bool): 是否在执行过程中输出详细日志。
    return_trace (bool): 是否在结果中保留中间执行痕迹。
''')

add_english_doc('ModuleTool', '''\
Base class for defining tools using callable Python functions.

This class automatically parses function signatures and docstrings to build a parameter schema using `pydantic`. It also performs input validation and handles standardized tool execution.

`__init__(self, verbose=False, return_trace=True)`
Initializes a tool wrapper module.

Args:
    verbose (bool): Whether to print verbose logs during execution.
    return_trace (bool): Whether to keep intermediate execution trace in the result.
''')

add_example('ModuleTool', """
>>> from lazyllm.components import ModuleTool
>>> class AddTool(ModuleTool):
...     def apply(self, a: int, b: int) -> int:
...         '''Add two integers.
...         
...         Args:
...             a (int): First number.
...             b (int): Second number.
...         
...         Returns:
...             int: The sum of a and b.
...         '''
...         return a + b
>>> tool = AddTool()
>>> result = tool({'a': 3, 'b': 5})
>>> print(result)
8
""")


add_chinese_doc('FunctionCall', '''\
FunctionCall是单轮工具调用类，如果LLM中的信息不足以回答用户的问题，必需结合外部知识来回答用户问题，则调用该类。如果LLM输出需要工具调用，则进行工具调用，并输出工具调用结果，输出结果为List类型，包含当前轮的输入、模型输出、工具输出。如果不需要工具调用，则直接输出LLM结果，输出结果为string类型。

Args:
    llm (ModuleBase): 要使用的LLM可以是TrainableModule或OnlineChatModule。
    tools (List[Union[str, Callable]]): LLM使用的工具名称或者 Callable 列表

注意：tools 中使用的工具必须带有 `__doc__` 字段，按照 [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) 的要求描述清楚工具的用途和参数。
''')

add_english_doc('FunctionCall', '''\
FunctionCall is a single-round tool call class. If the information in LLM is not enough to answer the uesr's question, it is necessary to combine external knowledge to answer the user's question. If the LLM output required a tool call, the tool call is performed and the tool call result is output. The output result is of List type, including the input, model output, and tool output of the current round. If a tool call is not required, the LLM result is directly output, and the output result is of string type.

Note: The tools used in `tools` must have a `__doc__` field, clearly describing the purpose and parameters of the tool according to the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) requirements.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[Union[str, Callable]]): A list of tool names for LLM to use.
''')

add_example('FunctionCall', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, FunctionCall
>>> import json
>>> from typing import Literal
>>> @fc_register("tool")
>>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = 'fahrenheit'):
...     '''
...     Get the current weather in a given location
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         unit (str): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> @fc_register("tool")
>>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
...     '''
...     Get an N-day weather forecast
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         num_days (int): The number of days to forecast.
...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> tools=["get_current_weather", "get_n_day_weather_forecast"]
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule("openai", stream=False)
>>> query = "What's the weather like today in celsius in Tokyo."
>>> fc = FunctionCall(llm, tools)
>>> ret = fc(query)
>>> print(ret)
["What's the weather like today in celsius in Tokyo.", {'role': 'assistant', 'content': '
', 'tool_calls': [{'id': 'da19cddac0584869879deb1315356d2a', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}}]}, [{'role': 'tool', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'tool_call_id': 'da19cddac0584869879deb1315356d2a', 'name': 'get_current_weather'}]]
>>> query = "Hello"
>>> ret = fc(query)
>>> print(ret)
'Hello! How can I assist you today?'
""")

add_chinese_doc('FunctionCallAgent', '''\
FunctionCallAgent是一个使用工具调用方式进行完整工具调用的代理，即回答用户问题时，LLM如果需要通过工具获取外部知识，就会调用工具，并将工具的返回结果反馈给LLM，最后由LLM进行汇总输出。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM 使用的工具名称列表。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
''')

add_english_doc('FunctionCallAgent', '''\
FunctionCallAgent is an agent that uses the tool calling method to perform complete tool calls. That is, when answering uesr questions, if LLM needs to obtain external knowledge through the tool, it will call the tool and feed back the return results of the tool to LLM, which will finally summarize and output them.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[str]): A list of tool names for LLM to use.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
''')

add_example('FunctionCallAgent', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, FunctionCallAgent
>>> import json
>>> from typing import Literal
>>> @fc_register("tool")
>>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
...     '''
...     Get the current weather in a given location
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         unit (str): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> @fc_register("tool")
>>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
...     '''
...     Get an N-day weather forecast
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         num_days (int): The number of days to forecast.
...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> tools = ['get_current_weather', 'get_n_day_weather_forecast']
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = FunctionCallAgent(llm, tools)
>>> query = "What's the weather like today in celsius in Tokyo and Paris."
>>> res = agent(query)
>>> print(res)
'The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.'
>>> query = "Hello"
>>> res = agent(query)
>>> print(res)
'Hello! How can I assist you today?'
""")

# actors/function_call_formatter.py
add_chinese_doc('FunctionCallFormatter', '''\
用于解析函数调用结构消息的格式化器。

该类继承自 `JsonFormatter`，用于从包含工具调用信息的消息字符串中提取 JSON 结构，并在需要时通过全局分隔符拆分内容。

私有方法:
    _load(msg)
        解析输入的消息字符串，提取其中的 JSON 格式的工具调用结构（如果存在）。
''')

add_english_doc('FunctionCallFormatter', '''\
Formatter for parsing structured function call messages.

This class extends `JsonFormatter` and is responsible for extracting JSON-based tool call structures from a mixed message string, optionally separating them using a global delimiter.

Private Method:
    _load(msg)
        Parses the input message string and extracts JSON-formatted tool calls, if present.
''')

add_example('FunctionCallFormatter', ['''\
>>> from lazyllm.components import FunctionCallFormatter
>>> formatter = FunctionCallFormatter()
>>> msg = "Please call this tool. <TOOL> [{\\"name\\": \\"search\\", \\"args\\": {\\"query\\": \\"weather\\"}}]"
>>> result = formatter._load(msg)
>>> print(result)
... [{'name': 'search', 'args': {'query': 'weather'}}, 'Please call this tool. ']
'''])

add_chinese_doc('FunctionCallFormatter', '''\
用于解析函数调用结构消息的格式化器。

该类继承自 `JsonFormatter`，用于从包含工具调用信息的消息字符串中提取 JSON 结构，并在需要时通过全局分隔符拆分内容。

私有方法:
    _load(msg)
        解析输入的消息字符串，提取其中的 JSON 格式的工具调用结构（如果存在）。
''')

add_english_doc('FunctionCallFormatter', '''\
Formatter for parsing structured function call messages.

This class extends `JsonFormatter` and is responsible for extracting JSON-based tool call structures from a mixed message string, optionally separating them using a global delimiter.

Private Method:
    _load(msg)
        Parses the input message string and extracts JSON-formatted tool calls, if present.
''')

add_example('FunctionCallFormatter', ['''\
>>> from lazyllm.components import FunctionCallFormatter
>>> formatter = FunctionCallFormatter()
>>> msg = "Please call this tool. <TOOL> [{\\"name\\": \\"search\\", \\"args\\": {\\"query\\": \\"weather\\"}}]"
>>> result = formatter._load(msg)
>>> print(result)
... [{'name': 'search', 'args': {'query': 'weather'}}, 'Please call this tool. ']
'''])

add_chinese_doc('ReactAgent', '''\
ReactAgent是按照 `Thought->Action->Observation->Thought...->Finish` 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM 使用的工具名称列表。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
''')

add_english_doc('ReactAgent', '''\
ReactAgent follows the process of `Thought->Action->Observation->Thought...->Finish` step by step through LLM and tool calls to display the steps to solve user questions and the final answer to the user.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[str]): A list of tool names for LLM to use.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
''')

add_example('ReactAgent', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, ReactAgent
>>> @fc_register("tool")
>>> def multiply_tool(a: int, b: int) -> int:
...     '''
...     Multiply two integers and return the result integer
...
...     Args:
...         a (int): multiplier
...         b (int): multiplier
...     '''
...     return a * b
...
>>> @fc_register("tool")
>>> def add_tool(a: int, b: int):
...     '''
...     Add two integers and returns the result integer
...
...     Args:
...         a (int): addend
...         b (int): addend
...     '''
...     return a + b
...
>>> tools = ["multiply_tool", "add_tool"]
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()   # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = ReactAgent(llm, tools)
>>> query = "What is 20+(2*4)? Calculate step by step."
>>> res = agent(query)
>>> print(res)
'Answer: The result of 20+(2*4) is 28.'
""")

add_chinese_doc('PlanAndSolveAgent', '''\
PlanAndSolveAgent由两个组件组成，首先，由planner将整个任务分解为更小的子任务，然后由solver根据计划执行这些子任务，其中可能会涉及到工具调用，最后将答案返回给用户。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
    tools (List[str]): LLM使用的工具名称列表。
    plan_llm (ModuleBase): planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
    solve_llm (ModuleBase): solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
''')

add_english_doc('PlanAndSolveAgent', '''\
PlanAndSolveAgent consists of two components. First, the planner breaks down the entire task into smaller subtasks, then the solver executes these subtasks according to the plan, which may involve tool calls, and finally returns the answer to the user.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and sovler share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
''')

add_example('PlanAndSolveAgent', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, PlanAndSolveAgent
>>> @fc_register("tool")
>>> def multiply(a: int, b: int) -> int:
...     '''
...     Multiply two integers and return the result integer
...
...     Args:
...         a (int): multiplier
...         b (int): multiplier
...     '''
...     return a * b
...
>>> @fc_register("tool")
>>> def add(a: int, b: int):
...     '''
...     Add two integers and returns the result integer
...
...     Args:
...         a (int): addend
...         b (int): addend
...     '''
...     return a + b
...
>>> tools = ["multiply", "add"]
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = PlanAndSolveAgent(llm, tools)
>>> query = "What is 20+(2*4)? Calculate step by step."
>>> res = agent(query)
>>> print(res)
'The final answer is 28.'
""")

add_chinese_doc('ReWOOAgent', '''\
ReWOOAgent包含三个部分：Planner、Worker和Solver。其中，Planner使用可预见推理能力为复杂任务创建解决方案蓝图；Worker通过工具调用来与环境交互，并将实际证据或观察结果填充到指令中；Solver处理所有计划和证据以制定原始任务或问题的解决方案。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
    tools (List[str]): LLM使用的工具名称列表。
    plan_llm (ModuleBase): planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
    solve_llm (ModuleBase): solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。

    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。

''')

add_english_doc('ReWOOAgent', '''\
ReWOOAgent consists of three parts: Planer, Worker and Solver. The Planner uses predictive reasoning capabilities to create a solution blueprint for a complex task; the Worker interacts with the environment through tool calls and fills in actual evidence or observations into instructions; the Solver processes all plans and evidence to develop a solution to the original task or problem.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and sovler share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
''')

add_example(
    "ReWOOAgent",
    """\
>>> import lazyllm
>>> import wikipedia
>>> from lazyllm.tools import fc_register, ReWOOAgent
>>> @fc_register("tool")
>>> def WikipediaWorker(input: str):
...     '''
...     Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.
...
...     Args:
...         input (str): search query.
...     '''
...     try:
...         evidence = wikipedia.page(input).content
...         evidence = evidence.split("\\\\n\\\\n")[0]
...     except wikipedia.PageError:
...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
...     except wikipedia.DisambiguationError:
...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
...     return evidence
...
>>> @fc_register("tool")
>>> def LLMWorker(input: str):
...     '''
...     A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.
...
...     Args:
...         input (str): instruction
...     '''
...     llm = lazyllm.OnlineChatModule(source="glm")
...     query = f"Respond in short directly with no extra words.\\\\n\\\\n{input}"
...     response = llm(query, llm_chat_history=[])
...     return response
...
>>> tools = ["WikipediaWorker", "LLMWorker"]
>>> llm = lazyllm.TrainableModule("GLM-4-9B-Chat").deploy_method(lazyllm.deploy.vllm).start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = ReWOOAgent(llm, tools)
>>> query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
>>> res = agent(query)
>>> print(res)
'\nHennessy '
""",
)

add_chinese_doc(
    "IntentClassifier",
    """\
IntentClassifier 是一个基于语言模型的意图识别器，用于根据用户提供的输入文本及对话上下文识别预定义的意图，并通过预处理和后处理步骤确保准确识别意图。

Arguments:
    llm: 用于意图识别的语言模型对象，OnlineChatModule或TrainableModule类型
    intent_list (list): 包含所有可能意图的字符串列表。可以包含中文或英文的意图。
    prompt (str): 用户附加的提示词。
    constrain (str): 用户附加的限制。
    examples (list[list]): 额外的示例，格式为 `[[query, intent], [query, intent], ...]` 。
    return_trace (bool, 可选): 如果设置为 True，则将结果记录在trace中。默认为 False。
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
    prompt (str): User-attached prompt words.
    constrain (str): User-attached constrain words.
    examples (list[list]): extra examples，format is `[[query, intent], [query, intent], ...]`.
    return_trace (bool, optional): If set to True, the results will be recorded in the trace. Defaults to False.
""",
)

add_example(
    "IntentClassifier",
    """\
    >>> import lazyllm
    >>> from lazyllm.tools import IntentClassifier
    >>> classifier_llm = lazyllm.OnlineChatModule(source="openai")
    >>> chatflow_intent_list = ["Chat", "Financial Knowledge Q&A", "Employee Information Query", "Weather Query"]
    >>> classifier = IntentClassifier(classifier_llm, intent_list=chatflow_intent_list)
    >>> classifier.start()
    >>> print(classifier('What is the weather today'))
    Weather Query
    >>>
    >>> with IntentClassifier(classifier_llm) as ic:
    >>>     ic.case['Weather Query', lambda x: '38.5°C']
    >>>     ic.case['Chat', lambda x: 'permission denied']
    >>>     ic.case['Financial Knowledge Q&A', lambda x: 'Calling Financial RAG']
    >>>     ic.case['Employee Information Query', lambda x: 'Beijing']
    ...
    >>> ic.start()
    >>> print(ic('What is the weather today'))
    38.5°C
""",
)

#eval/eval_base.py
add_chinese_doc('BaseEvaluator', '''\
评估模块的抽象基类。

该类定义了模型评估的标准接口，支持并发处理、输入校验和评估结果的自动保存，同时内置了重试机制。

Args:
    concurrency (int): 评估过程中使用的并发线程数。
    retry (int): 每个样本的最大重试次数。
    log_base_name (Optional[str]): 用于保存结果文件的日志文件名前缀（可选）。
''')

add_english_doc('BaseEvaluator', '''\
Abstract base class for evaluation modules.

This class defines the standard interface and retry logic for evaluating model outputs. It supports concurrent processing, input validation, and automatic result saving.

Args:
    concurrency (int): Number of concurrent threads used during evaluation.
    retry (int): Number of retry attempts for each evaluation item.
    log_base_name (Optional[str]): Optional log file name prefix for saving results.
''')

add_example('BaseEvaluator', ['''\
>>> from lazyllm.components import BaseEvaluator
>>> class SimpleAccuracyEvaluator(BaseEvaluator):
...     def _process_one_data_impl(self, data):
...         return {
...             "final_score": float(data["pred"] == data["label"])
...         }
>>> evaluator = SimpleAccuracyEvaluator()
>>> score = evaluator([
...     {"pred": "yes", "label": "yes"},
...     {"pred": "no", "label": "yes"}
... ])
>>> print(score)
... 0.5
'''])

add_chinese_doc('ResponseRelevancy', '''\
用于评估用户问题与模型生成问题之间语义相关性的指标类。

该评估器使用语言模型根据回答生成问题，并通过 Embedding 与余弦相似度度量其与原始问题之间的相关性。


Args:
    llm (ModuleBase): 用于根据回答生成问题的语言模型模块。
    embedding (ModuleBase): 用于编码问题向量的嵌入模块。
    prompt (str, 可选): 自定义的生成提示词，若不提供将使用默认提示。
    prompt_lang (str): 默认提示词的语言，可选 `'en'`（默认）或 `'zh'`。
    num_infer_questions (int): 每条数据生成和评估的问题数量。
    retry (int): 失败时的重试次数。
    concurrency (int): 并发评估的数量。
''')

add_english_doc('ResponseRelevancy', '''\
Evaluator for measuring the semantic relevancy between a user-generated question and a model-generated one.

This evaluator uses a language model to generate possible questions from an answer, and measures their semantic similarity to the original question using embeddings and cosine similarity.


Args:
    llm (ModuleBase): A language model used to generate inferred questions from the given answer.
    embedding (ModuleBase): An embedding module to encode questions for similarity comparison.
    prompt (str, optional): Custom prompt to guide the question generation. If not provided, a default will be used.
    prompt_lang (str): Language for the default prompt. Options: `'en'` (default) or `'zh'`.
    num_infer_questions (int): Number of questions to generate and evaluate for each answer.
    retry (int): Number of retry attempts if generation fails.
    concurrency (int): Number of concurrent evaluations.
''')

add_example('ResponseRelevancy', ['''\
>>> from lazyllm.components import ResponseRelevancy
>>> relevancy = ResponseRelevancy(
...     llm=YourLLM(),
...     embedding=YourEmbedding(),
...     prompt_lang="en",
...     num_infer_questions=3
... )
>>> result = relevancy([
...     {"question": "What is the capital of France?", "answer": "Paris is the capital city of France."}
... ])
>>> print(result)
... 0.95  # (a float score between 0 and 1)
'''])

add_chinese_doc('Faithfulness', '''\
评估回答与上下文之间事实一致性的指标类。

该评估器首先使用语言模型将答案拆分为独立事实句，然后基于上下文对每条句子进行支持性判断（0或1分），最终取平均值作为总体一致性分数。


Args:
    llm (ModuleBase): 同时用于生成句子与进行评估的语言模型模块。
    generate_prompt (str, 可选): 用于将答案转换为事实句的自定义提示词。
    eval_prompt (str, 可选): 用于评估句子与上下文匹配度的提示词。
    prompt_lang (str): 默认提示词的语言，可选 'en' 或 'zh'。
    retry (int): 生成或评估失败时的最大重试次数。
    concurrency (int): 并发评估的数据条数。
''')

add_english_doc('Faithfulness', '''\
Evaluator that measures the factual consistency of an answer with the given context.

This evaluator splits the answer into atomic factual statements using a generation model, then verifies each against the context using binary (1/0) scoring. It computes a final score as the average of the individual statement scores.


Args:
    llm (ModuleBase): A language model capable of both generating statements and evaluating them.
    generate_prompt (str, optional): Custom prompt to generate factual statements from the answer.
    eval_prompt (str, optional): Custom prompt to evaluate statement support within the context.
    prompt_lang (str): Language of the default prompt, either 'en' or 'zh'.
    retry (int): Number of retry attempts when generation or evaluation fails.
    concurrency (int): Number of concurrent evaluations to run in parallel.
''')

add_example('Faithfulness', ['''\
>>> from lazyllm.components import Faithfulness
>>> evaluator = Faithfulness(llm=YourLLM(), prompt_lang="en")
>>> data = {
...     "question": "What is the role of ATP in cells?",
...     "answer": "ATP stores energy and transfers it within cells.",
...     "context": "ATP is the energy currency of the cell. It provides energy for many biochemical reactions."
... }
>>> result = evaluator([data])
>>> print(result)
... 1.0  # Average binary score of all factual statements
'''])

add_chinese_doc('LLMContextRecall', '''\
用于评估回答中的每一句话是否可以归因于检索到的上下文的指标类。

该模块使用语言模型判断回答中的每个句子是否得到上下文的支持，通过二元值进行评分（1 表示支持，0 表示不支持或矛盾），最终计算平均回忆得分。

Args:
    llm (ModuleBase): 用于执行上下文一致性判断的语言模型。
    eval_prompt (str, 可选): 指导模型评估的自定义提示词。
    prompt_lang (str): 默认提示词语言，'en' 表示英文，'zh' 表示中文。
    retry (int): 评估失败时的最大重试次数。
    concurrency (int): 并发评估的任务数量。
''')

add_english_doc('LLMContextRecall', '''\
Evaluator that measures whether each sentence in the answer can be attributed to the retrieved context.

This module uses a language model to analyze the factual alignment between each statement in the answer and the provided context. It scores each sentence with binary values (1 = supported, 0 = unsupported/contradictory) and computes an average recall score.


Args:
    llm (ModuleBase): A language model capable of evaluating answer-context consistency.
    eval_prompt (str, optional): Custom prompt used to instruct the evaluator model.
    prompt_lang (str): Language of the default prompt. Choose 'en' for English or 'zh' for Chinese.
    retry (int): Number of retry attempts if the evaluation fails.
    concurrency (int): Number of parallel evaluations to perform concurrently.
''')

add_example('LLMContextRecall', ['''\
>>> from lazyllm.components import LLMContextRecall
>>> evaluator = LLMContextRecall(llm=YourLLM(), prompt_lang="en")
>>> data = {
...     "question": "What is Photosynthesis?",
...     "answer": "Photosynthesis was discovered in the 1780s. It occurs in chloroplasts.",
...     "context_retrieved": [
...         "Photosynthesis occurs in chloroplasts.",
...         "Light reactions produce ATP using sunlight."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 0.5  # Final recall score averaged over statement evaluations
'''])

add_chinese_doc('NonLLMContextRecall', '''\
基于字符串模糊匹配的非LLM上下文回忆指标类。

该模块通过 Levenshtein 距离计算检索到的上下文与参考上下文的相似度，并给出回忆得分。可选择输出二值得分（是否存在足够相似的匹配）或平均匹配度得分。

Args:
    th (float): 相似度阈值（范围为0到1），值越高表示匹配越严格。
    binary (bool): 若为True，则只判断是否有任一匹配超过阈值；若为False，则输出所有匹配的平均得分。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。
''')

add_english_doc('NonLLMContextRecall', '''\
A non-LLM evaluator that measures whether retrieved contexts match the reference context using fuzzy string matching.

This module compares each retrieved context against a reference using Levenshtein distance and computes a recall score. It can return binary scores (whether any retrieved context is similar enough) or an averaged similarity score.


Args:
    th (float): Similarity threshold (between 0 and 1). A higher value means stricter matching.
    binary (bool): If True, output is binary (1 if any match exceeds threshold), otherwise returns average match score.
    retry (int): Number of retries for evaluation in case of failure.
    concurrency (int): Number of parallel evaluations to run.
''')

add_example('NonLLMContextRecall', ['''\
>>> from lazyllm.components import NonLLMContextRecall
>>> evaluator = NonLLMContextRecall(th=0.8, binary=True)
>>> data = {
...     "context_retrieved": [
...         "Photosynthesis uses sunlight to produce sugar.",
...         "It takes place in chloroplasts."
...     ],
...     "context_reference": [
...         "Photosynthesis occurs in chloroplasts."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 1.0  # At least one retrieved context is similar enough
'''])

add_chinese_doc('ContextRelevance', '''\
基于句子级匹配的非LLM上下文相关性评估器。

该模块将检索到的上下文与参考上下文分别按句子划分，并统计检索内容中与参考完全一致的句子数量，从而计算相关性得分。


Args:
    splitter (str): 句子分隔符，默认为中文句号 "。"，英文可设置为 "."。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。
''')

add_english_doc('ContextRelevance', '''\
A non-LLM evaluator that measures the overlap between retrieved and reference contexts at the sentence level.

This evaluator splits both retrieved and reference contexts into sentences, then counts how many retrieved sentences exactly match those in the reference. It outputs a relevance score as the fraction of overlapping sentences.


Args:
    splitter (str): Sentence splitter. Default is '。' for Chinese. Use '.' for English contexts.
    retry (int): Number of retries for evaluation in case of failure.
    concurrency (int): Number of parallel evaluations to run.
''')

add_example('ContextRelevance', ['''\
>>> from lazyllm.components import ContextRelevance
>>> evaluator = ContextRelevance(splitter='.')
>>> data = {
...     "context_retrieved": [
...         "Photosynthesis occurs in chloroplasts. It produces glucose."
...     ],
...     "context_reference": [
...         "Photosynthesis occurs in chloroplasts. It requires sunlight. It produces glucose."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 0.6667  # 2 of 3 retrieved sentences match
'''])



#http_request/http_request.py
add_chinese_doc('HttpRequest', '''\
通用 HTTP 请求执行器。

该类用于构建并发送 HTTP 请求，支持变量替换、API Key 注入、JSON 或表单编码、文件类型响应识别等功能。

Args:
    method (str): HTTP 方法，如 'GET'、'POST' 等。
    url (str): 请求目标的 URL。
    api_key (str): 可选的 API Key，会被加入请求参数。
    headers (dict): HTTP 请求头。
    params (dict): URL 查询参数。
    body (Union[str, dict]): 请求体，支持字符串或 JSON 字典格式。
    timeout (int): 请求超时时间（秒）。
    proxies (dict, optional): 可选的代理设置。
''')

add_english_doc('HttpRequest', '''\
General HTTP request executor.

This class builds and sends HTTP requests with support for dynamic variable substitution, API key injection, JSON or form data encoding, and file-aware response parsing.

Args:
    method (str): HTTP method, such as 'GET', 'POST', etc.
    url (str): The target URL for the HTTP request.
    api_key (str): Optional API key, inserted into query parameters.
    headers (dict): HTTP request headers.
    params (dict): URL query parameters.
    body (Union[str, dict]): HTTP request body (raw string or JSON-formatted dict).
    timeout (int): Timeout duration for the request (in seconds).
    proxies (dict, optional): Proxy settings for the request, if needed.
''')

add_example('HttpRequest', ['''\
>>> from lazyllm.components import HttpRequest
>>> request = HttpRequest(
...     method="GET",
...     url="https://api.github.com/repos/openai/openai-python",
...     api_key="",
...     headers={"Accept": "application/json"},
...     params={},
...     body=None
... )
>>> result = request()
>>> print(result["status_code"])
... 200
>>> print(result["content"][:100])
... '{"id":123456,"name":"openai-python", ...}'
'''])

#infer_service/serve.py/JobDescription
add_chinese_doc('JobDescription', '''\
模型部署任务描述的数据结构。

用于创建模型推理任务时指定部署配置，包括模型名称与所需 GPU 数量。

Args:
    deploy_model (str): 要部署的模型名称，默认为 "qwen1.5-0.5b-chat"。
    num_gpus (int): 所需的 GPU 数量，默认为 1。
''')

add_english_doc('JobDescription', '''\
Model deployment job description schema.

Used to specify the configuration for creating a model inference job, including model name and GPU requirements.

Args:
    deploy_model (str): The model to be deployed. Default is "qwen1.5-0.5b-chat".
    num_gpus (int): Number of GPUs required for deployment. Default is 1.
''')

add_example('JobDescription', ['''\
>>> from lazyllm.components import JobDescription
>>> job = JobDescription(deploy_model="deepseek-coder", num_gpus=2)
>>> print(job.dict())
... {'deploy_model': 'deepseek-coder', 'num_gpus': 2}
'''])


add_chinese_doc('DBManager', '''\
数据库管理器的抽象基类。

该类定义了构建数据库连接器的通用接口，包括 `execute_query` 抽象方法和 `desc` 描述属性。

Args:
    db_type (str): 数据库类型标识符，例如 'mysql'、'mongodb'。
''')

add_english_doc('DBManager', '''\
Abstract base class for database managers.

This class defines the standard interface and helpers for building database connectors, including a required `execute_query` method and description property.

Args:
    db_type (str): Type identifier of the database (e.g., 'mysql', 'mongodb').
''')

add_example('DBManager', ['''\
>>> from lazyllm.components import DBManager
>>> class DummyDB(DBManager):
...     def __init__(self):
...         super().__init__(db_type="dummy")
...     def execute_query(self, statement):
...         return f"Executed: {statement}"
...     @property
...     def desc(self):
...         return "Dummy database for testing."
>>> db = DummyDB()
>>> print(db("SELECT * FROM test"))
... Executed: SELECT * FROM test
'''])

add_chinese_doc('BaseEvaluator', '''\
评估模块的抽象基类。

该类定义了模型评估的标准接口，支持并发处理、输入校验和评估结果的自动保存，同时内置了重试机制。

Args:
    concurrency (int): 评估过程中使用的并发线程数。
    retry (int): 每个样本的最大重试次数。
    log_base_name (Optional[str]): 用于保存结果文件的日志文件名前缀（可选）。
''')

add_english_doc('BaseEvaluator', '''\
Abstract base class for evaluation modules.

This class defines the standard interface and retry logic for evaluating model outputs. It supports concurrent processing, input validation, and automatic result saving.

Args:
    concurrency (int): Number of concurrent threads used during evaluation.
    retry (int): Number of retry attempts for each evaluation item.
    log_base_name (Optional[str]): Optional log file name prefix for saving results.
''')

add_example('BaseEvaluator', ['''\
>>> from lazyllm.components import BaseEvaluator
>>> class SimpleAccuracyEvaluator(BaseEvaluator):
...     def _process_one_data_impl(self, data):
...         return {
...             "final_score": float(data["pred"] == data["label"])
...         }
>>> evaluator = SimpleAccuracyEvaluator()
>>> score = evaluator([
...     {"pred": "yes", "label": "yes"},
...     {"pred": "no", "label": "yes"}
... ])
>>> print(score)
... 0.5
'''])

add_chinese_doc('ResponseRelevancy', '''\
用于评估用户问题与模型生成问题之间语义相关性的指标类。

该评估器使用语言模型根据回答生成问题，并通过 Embedding 与余弦相似度度量其与原始问题之间的相关性。


Args:
    llm (ModuleBase): 用于根据回答生成问题的语言模型模块。
    embedding (ModuleBase): 用于编码问题向量的嵌入模块。
    prompt (str, 可选): 自定义的生成提示词，若不提供将使用默认提示。
    prompt_lang (str): 默认提示词的语言，可选 `'en'`（默认）或 `'zh'`。
    num_infer_questions (int): 每条数据生成和评估的问题数量。
    retry (int): 失败时的重试次数。
    concurrency (int): 并发评估的数量。
''')

add_english_doc('ResponseRelevancy', '''\
Evaluator for measuring the semantic relevancy between a user-generated question and a model-generated one.

This evaluator uses a language model to generate possible questions from an answer, and measures their semantic similarity to the original question using embeddings and cosine similarity.


Args:
    llm (ModuleBase): A language model used to generate inferred questions from the given answer.
    embedding (ModuleBase): An embedding module to encode questions for similarity comparison.
    prompt (str, optional): Custom prompt to guide the question generation. If not provided, a default will be used.
    prompt_lang (str): Language for the default prompt. Options: `'en'` (default) or `'zh'`.
    num_infer_questions (int): Number of questions to generate and evaluate for each answer.
    retry (int): Number of retry attempts if generation fails.
    concurrency (int): Number of concurrent evaluations.
''')

add_example('ResponseRelevancy', ['''\
>>> from lazyllm.components import ResponseRelevancy
>>> relevancy = ResponseRelevancy(
...     llm=YourLLM(),
...     embedding=YourEmbedding(),
...     prompt_lang="en",
...     num_infer_questions=3
... )
>>> result = relevancy([
...     {"question": "What is the capital of France?", "answer": "Paris is the capital city of France."}
... ])
>>> print(result)
... 0.95  # (a float score between 0 and 1)
'''])

add_chinese_doc('Faithfulness', '''\
评估回答与上下文之间事实一致性的指标类。

该评估器首先使用语言模型将答案拆分为独立事实句，然后基于上下文对每条句子进行支持性判断（0或1分），最终取平均值作为总体一致性分数。


Args:
    llm (ModuleBase): 同时用于生成句子与进行评估的语言模型模块。
    generate_prompt (str, 可选): 用于将答案转换为事实句的自定义提示词。
    eval_prompt (str, 可选): 用于评估句子与上下文匹配度的提示词。
    prompt_lang (str): 默认提示词的语言，可选 'en' 或 'zh'。
    retry (int): 生成或评估失败时的最大重试次数。
    concurrency (int): 并发评估的数据条数。
''')

add_english_doc('Faithfulness', '''\
Evaluator that measures the factual consistency of an answer with the given context.

This evaluator splits the answer into atomic factual statements using a generation model, then verifies each against the context using binary (1/0) scoring. It computes a final score as the average of the individual statement scores.


Args:
    llm (ModuleBase): A language model capable of both generating statements and evaluating them.
    generate_prompt (str, optional): Custom prompt to generate factual statements from the answer.
    eval_prompt (str, optional): Custom prompt to evaluate statement support within the context.
    prompt_lang (str): Language of the default prompt, either 'en' or 'zh'.
    retry (int): Number of retry attempts when generation or evaluation fails.
    concurrency (int): Number of concurrent evaluations to run in parallel.
''')

add_example('Faithfulness', ['''\
>>> from lazyllm.components import Faithfulness
>>> evaluator = Faithfulness(llm=YourLLM(), prompt_lang="en")
>>> data = {
...     "question": "What is the role of ATP in cells?",
...     "answer": "ATP stores energy and transfers it within cells.",
...     "context": "ATP is the energy currency of the cell. It provides energy for many biochemical reactions."
... }
>>> result = evaluator([data])
>>> print(result)
... 1.0  # Average binary score of all factual statements
'''])

add_chinese_doc('LLMContextRecall', '''\
用于评估回答中的每一句话是否可以归因于检索到的上下文的指标类。

该模块使用语言模型判断回答中的每个句子是否得到上下文的支持，通过二元值进行评分（1 表示支持，0 表示不支持或矛盾），最终计算平均回忆得分。

Args:
    llm (ModuleBase): 用于执行上下文一致性判断的语言模型。
    eval_prompt (str, 可选): 指导模型评估的自定义提示词。
    prompt_lang (str): 默认提示词语言，'en' 表示英文，'zh' 表示中文。
    retry (int): 评估失败时的最大重试次数。
    concurrency (int): 并发评估的任务数量。
''')

add_english_doc('LLMContextRecall', '''\
Evaluator that measures whether each sentence in the answer can be attributed to the retrieved context.

This module uses a language model to analyze the factual alignment between each statement in the answer and the provided context. It scores each sentence with binary values (1 = supported, 0 = unsupported/contradictory) and computes an average recall score.


Args:
    llm (ModuleBase): A language model capable of evaluating answer-context consistency.
    eval_prompt (str, optional): Custom prompt used to instruct the evaluator model.
    prompt_lang (str): Language of the default prompt. Choose 'en' for English or 'zh' for Chinese.
    retry (int): Number of retry attempts if the evaluation fails.
    concurrency (int): Number of parallel evaluations to perform concurrently.
''')

add_example('LLMContextRecall', ['''\
>>> from lazyllm.components import LLMContextRecall
>>> evaluator = LLMContextRecall(llm=YourLLM(), prompt_lang="en")
>>> data = {
...     "question": "What is Photosynthesis?",
...     "answer": "Photosynthesis was discovered in the 1780s. It occurs in chloroplasts.",
...     "context_retrieved": [
...         "Photosynthesis occurs in chloroplasts.",
...         "Light reactions produce ATP using sunlight."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 0.5  # Final recall score averaged over statement evaluations
'''])

add_chinese_doc('NonLLMContextRecall', '''\
基于字符串模糊匹配的非LLM上下文回忆指标类。

该模块通过 Levenshtein 距离计算检索到的上下文与参考上下文的相似度，并给出回忆得分。可选择输出二值得分（是否存在足够相似的匹配）或平均匹配度得分。

Args:
    th (float): 相似度阈值（范围为0到1），值越高表示匹配越严格。
    binary (bool): 若为True，则只判断是否有任一匹配超过阈值；若为False，则输出所有匹配的平均得分。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。
''')

add_english_doc('NonLLMContextRecall', '''\
A non-LLM evaluator that measures whether retrieved contexts match the reference context using fuzzy string matching.

This module compares each retrieved context against a reference using Levenshtein distance and computes a recall score. It can return binary scores (whether any retrieved context is similar enough) or an averaged similarity score.


Args:
    th (float): Similarity threshold (between 0 and 1). A higher value means stricter matching.
    binary (bool): If True, output is binary (1 if any match exceeds threshold), otherwise returns average match score.
    retry (int): Number of retries for evaluation in case of failure.
    concurrency (int): Number of parallel evaluations to run.
''')

add_example('NonLLMContextRecall', ['''\
>>> from lazyllm.components import NonLLMContextRecall
>>> evaluator = NonLLMContextRecall(th=0.8, binary=True)
>>> data = {
...     "context_retrieved": [
...         "Photosynthesis uses sunlight to produce sugar.",
...         "It takes place in chloroplasts."
...     ],
...     "context_reference": [
...         "Photosynthesis occurs in chloroplasts."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 1.0  # At least one retrieved context is similar enough
'''])

add_chinese_doc('ContextRelevance', '''\
基于句子级匹配的非LLM上下文相关性评估器。

该模块将检索到的上下文与参考上下文分别按句子划分，并统计检索内容中与参考完全一致的句子数量，从而计算相关性得分。


Args:
    splitter (str): 句子分隔符，默认为中文句号 "。"，英文可设置为 "."。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。
''')

add_english_doc('ContextRelevance', '''\
A non-LLM evaluator that measures the overlap between retrieved and reference contexts at the sentence level.

This evaluator splits both retrieved and reference contexts into sentences, then counts how many retrieved sentences exactly match those in the reference. It outputs a relevance score as the fraction of overlapping sentences.


Args:
    splitter (str): Sentence splitter. Default is '。' for Chinese. Use '.' for English contexts.
    retry (int): Number of retries for evaluation in case of failure.
    concurrency (int): Number of parallel evaluations to run.
''')

add_example('ContextRelevance', ['''\
>>> from lazyllm.components import ContextRelevance
>>> evaluator = ContextRelevance(splitter='.')
>>> data = {
...     "context_retrieved": [
...         "Photosynthesis occurs in chloroplasts. It produces glucose."
...     ],
...     "context_reference": [
...         "Photosynthesis occurs in chloroplasts. It requires sunlight. It produces glucose."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 0.6667  # 2 of 3 retrieved sentences match
'''])

add_chinese_doc('HttpRequest', '''\
通用 HTTP 请求执行器。

该类用于构建并发送 HTTP 请求，支持变量替换、API Key 注入、JSON 或表单编码、文件类型响应识别等功能。

Args:
    method (str): HTTP 方法，如 'GET'、'POST' 等。
    url (str): 请求目标的 URL。
    api_key (str): 可选的 API Key，会被加入请求参数。
    headers (dict): HTTP 请求头。
    params (dict): URL 查询参数。
    body (Union[str, dict]): 请求体，支持字符串或 JSON 字典格式。
    timeout (int): 请求超时时间（秒）。
    proxies (dict, optional): 可选的代理设置。
''')

add_english_doc('HttpRequest', '''\
General HTTP request executor.

This class builds and sends HTTP requests with support for dynamic variable substitution, API key injection, JSON or form data encoding, and file-aware response parsing.

Args:
    method (str): HTTP method, such as 'GET', 'POST', etc.
    url (str): The target URL for the HTTP request.
    api_key (str): Optional API key, inserted into query parameters.
    headers (dict): HTTP request headers.
    params (dict): URL query parameters.
    body (Union[str, dict]): HTTP request body (raw string or JSON-formatted dict).
    timeout (int): Timeout duration for the request (in seconds).
    proxies (dict, optional): Proxy settings for the request, if needed.
''')

add_example('HttpRequest', ['''\
>>> from lazyllm.components import HttpRequest
>>> request = HttpRequest(
...     method="GET",
...     url="https://api.github.com/repos/openai/openai-python",
...     api_key="",
...     headers={"Accept": "application/json"},
...     params={},
...     body=None
... )
>>> result = request()
>>> print(result["status_code"])
... 200
>>> print(result["content"][:100])
... '{"id":123456,"name":"openai-python", ...}'
'''])

add_chinese_doc('HttpExecutorResponse', '''\
HTTP 响应封装类。

该类封装了 httpx.Response 对象，提供了访问响应头、正文、状态码、内容类型等的便捷接口，并支持识别文件类型响应和提取文件。

Args:
    response (httpx.Response, optional): 可选的 httpx 响应对象。
''')

add_english_doc('HttpExecutorResponse', '''\
Wrapper for HTTP response.

This class wraps an httpx.Response object and provides convenient access to headers, body, status code, content type, and file-type response recognition and extraction.

Args:
    response (httpx.Response, optional): Optional HTTP response object from httpx.
''')

add_example('HttpExecutorResponse', ['''\
>>> import httpx
>>> from lazyllm.components import HttpExecutorResponse
>>> resp = httpx.Response(200, headers={"Content-Type": "application/json"}, content=b'{"msg":"hello"}')
>>> wrapper = HttpExecutorResponse(resp)
>>> print(wrapper.status_code)
... 200
>>> print(wrapper.content)
... {"msg":"hello"}
>>> print(wrapper.is_file)
... False
>>> print(wrapper.extract_file())
... ('', b'')
'''])

add_chinese_doc('JobDescription', '''\
模型部署任务描述的数据结构。

用于创建模型推理任务时指定部署配置，包括模型名称与所需 GPU 数量。

Args:
    deploy_model (str): 要部署的模型名称，默认为 "qwen1.5-0.5b-chat"。
    num_gpus (int): 所需的 GPU 数量，默认为 1。
''')

add_english_doc('JobDescription', '''\
Model deployment job description schema.

Used to specify the configuration for creating a model inference job, including model name and GPU requirements.

Args:
    deploy_model (str): The model to be deployed. Default is "qwen1.5-0.5b-chat".
    num_gpus (int): Number of GPUs required for deployment. Default is 1.
''')

add_example('JobDescription', ['''\
>>> from lazyllm.components import JobDescription
>>> job = JobDescription(deploy_model="deepseek-coder", num_gpus=2)
>>> print(job.dict()) 
... {'deploy_model': 'deepseek-coder', 'num_gpus': 2}
'''])


add_chinese_doc('DBManager', '''\
数据库管理器的抽象基类。

该类定义了构建数据库连接器的通用接口，包括 `execute_query` 抽象方法和 `desc` 描述属性。

Args:
    db_type (str): 数据库类型标识符，例如 'mysql'、'mongodb'。
''')

add_english_doc('DBManager', '''\
Abstract base class for database managers.

This class defines the standard interface and helpers for building database connectors, including a required `execute_query` method and description property.

Args:
    db_type (str): Type identifier of the database (e.g., 'mysql', 'mongodb').
''')

add_example('DBManager', ['''\
>>> from lazyllm.components import DBManager
>>> class DummyDB(DBManager):
...     def __init__(self):
...         super().__init__(db_type="dummy")
...     def execute_query(self, statement):
...         return f"Executed: {statement}"
...     @property
...     def desc(self):
...         return "Dummy database for testing."
>>> db = DummyDB()
>>> print(db("SELECT * FROM test"))
... Executed: SELECT * FROM test
'''])

add_chinese_doc(
    "SqlManager",
    """\
SqlManager是与数据库进行交互的专用工具。它提供了连接数据库，设置、创建、检查数据表，插入数据，执行查询的方法。

Arguments:
    db_type (str): "PostgreSQL"，"SQLite", "MySQL", "MSSQL"。注意当类型为"SQLite"时，db_name为文件路径或者":memory:"
    user (str): 用户名
    password (str): 密码
    host (str): 主机名或IP
    port (int): 端口号
    db_name (str): 数据仓库名
    **options_str (str): k1=v1&k2=v2形式表示的选项设置
""",
)

add_english_doc(
    "SqlManager",
    """\
SqlManager is a specialized tool for interacting with databases.
It provides methods for creating tables, executing queries, and performing updates on databases.

Arguments:
    db_type (str): "PostgreSQL"，"SQLite", "MySQL", "MSSQL". Note that when the type is "SQLite", db_name is a file path or ":memory:"
    user (str): Username for connection
    password (str): Password for connection
    host (str): Hostname or IP
    port (int): Port number
    db_name (str): Name of the database
    **options_str (str): Options represented in the format k1=v1&k2=v2
""",
)

add_chinese_doc(
    "SqlManager.get_session",
    """\
这是一个上下文管理器，它创建并返回一个数据库连接Session，并在完成时自动提交或回滚更改并在使用完成后自动关闭会话。

**Returns:**\n
- sqlalchemy.orm.Session: sqlalchemy 数据库会话
""",
)

add_english_doc(
    "SqlManager.get_session",
    """\
This is a context manager that creates and returns a database session, yields it for use, and then automatically commits or rolls back changes and closes the session when done.

**Returns:**\n
- sqlalchemy.orm.Session: sqlalchemy database session
""",
)

add_chinese_doc(
    "SqlManager.check_connection",
    """\
检查当前SqlManager的连接状态。

**Returns:**\n
- DBResult: DBResult.status 连接成功(True), 连接失败(False)。DBResult.detail 包含失败信息
""",
)

add_english_doc(
    "SqlManager.check_connection",
    """\
Check the current connection status of the SqlManagerBase.

**Returns:**\n
- DBResult: DBResult.status True if the connection is successful, False if it fails. DBResult.detail contains failure information.
""",
)

add_chinese_doc(
    "SqlManager.set_desc",
    """\
对于SqlManager搭配LLM使用自然语言查询的表项设置其描述，尤其当其表名、列名及取值不具有自解释能力时。
例如：
数据表Document的status列取值包括: "waiting", "working", "success", "failed"，tables_desc_dict参数应为 {"Document": "status列取值包括: waiting, working, success, failed"}

Args:
    tables_desc_dict (dict): 表项的补充说明
""",
)

add_english_doc(
    "SqlManager.set_desc",
    """\
When using SqlManager with LLM to query table entries in natural language, set descriptions for better results, especially when table names, column names, and values are not self-explanatory.

Args:
    tables_desc_dict (dict): descriptive comment for tables
""",
)

add_chinese_doc(
    "SqlManager.get_all_tables",
    """\
返回当前数据库中的所有表名。
""",
)

add_english_doc(
    "SqlManager.get_all_tables",
    """\
Return all table names in the current database.
""",
)

add_chinese_doc(
    "SqlManager.get_table_orm_class",
    """\
返回数据表名对应的sqlalchemy orm类。结合get_session，进行orm操作
""",
)

add_english_doc(
    "SqlManager.get_table_orm_class",
    """\
Return the sqlalchemy orm class corresponding to the given table name. Combine with get_session to perform orm operations.
""",
)

add_chinese_doc(
    "SqlManager.execute_commit",
    """\
执行无返回的sql脚本并提交更改。
""",
)

add_english_doc(
    "SqlManager.execute_commit",
    """\
Execute the SQL script without return and submit changes.
""",
)

add_chinese_doc(
    "SqlManager.execute_query",
    """\
执行sql查询脚本并以JSON字符串返回结果。
""",
)

add_english_doc(
    "SqlManager.execute_query",
    """\
Execute the SQL query script and return the result as a JSON string.
""",
)

add_chinese_doc(
    "SqlManager.create_table",
    """\
创建数据表

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): 数据表schema。支持三种参数类型：类型为str的sql语句，继承自DeclarativeBase或继承自declarative_base()的ORM类
""",
)

add_english_doc(
    "SqlManager.create_table",
    """\
Create a table

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): table schema。Supports three types of parameters: SQL statements with type str, ORM classes that inherit from DeclarativeBase or declarative_base().
""",
)

add_chinese_doc(
    "SqlManager.drop_table",
    """\
删除数据表

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): 数据表schema。支持三种参数类型：类型为str的数据表名，继承自DeclarativeBase或继承自declarative_base()的ORM类
""",
)

add_english_doc(
    "SqlManager.drop_table",
    """\
Delete a table

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): table schema。Supports three types of parameters: Table name with type str, ORM classes that inherit from DeclarativeBase or declarative_base().
""",
)

add_chinese_doc(
    "SqlManager.insert_values",
    """\
批量数据插入

Args:
    table_name (str): 数据表名
    vals (List[dict]): 待插入数据，格式为[{"col_name1": v01, "col_name2": v02, ...}, {"col_name1": v11, "col_name2": v12, ...}, ...]
""",
)

add_english_doc(
    "SqlManager.insert_values",
    """\
Bulk insert data

Args:
    table_name (str): Table name
    vals (List[dict]): data to be inserted, format as [{"col_name1": v01, "col_name2": v02, ...}, {"col_name1": v11, "col_name2": v12, ...}, ...]
""",
)

add_chinese_doc(
    "MongoDBManager",
    """\
MongoDBManager是与MongoB数据库进行交互的专用工具。它提供了检查连接，获取数据库连接对象，执行查询的方法。

Arguments:
    user (str): 用户名
    password (str): 密码
    host (str): 主机名或IP
    port (int): 端口号
    db_name (str): 数据仓库名
    collection_name (str): 集合名
    **options_str (str): k1=v1&k2=v2形式表示的选项设置
    **collection_desc_dict (dict): 集合内文档关键字描述，默认为空。不同于关系型数据库行和列的概念，MongoDB集合中的文档可以有完全不同的关键字，因此当配合LLM进行自然语言查询时需要提供必须的关键字的描述以获得更好的结果。
""",
)

add_english_doc(
    "MongoDBManager",
    """\
MongoDBManager is a specialized tool for interacting with MongoB databases.
It provides methods to check the connection, obtain the database connection object, and execute query.

Arguments:
    user (str): Username for connection
    password (str): Password for connection
    host (str): Hostname or IP
    port (int): Port number
    db_name (str): Name of the database
    collection_name (str): Name of the collection
    **options_str (str): Options represented in the format k1=v1&k2=v2
    **collection_desc_dict (dict): Document keyword description in the collection, which is None by default. Different from the concept of rows and columns in relational databases, documents in MongoDB collections can have completely different keywords. Therefore, when using LLM to perform natural language queries, it is necessary to provide descriptions of necessary keywords to obtain better results.
""",
)

add_example('MongoDBManager', ['''\
>>> from lazyllm.components import MongoDBManager
>>> mgr = MongoDBManager(
...     user="admin",
...     password="123456",
...     host="localhost",
...     port=27017,
...     db_name="mydb",
...     collection_name="books"
... )
>>> result = mgr.execute_query('[{"$match": {"author": "Tolstoy"}}]')
>>> print(result)
... '[{"title": "War and Peace", "author": "Tolstoy"}]'
'''])


add_example('MongoDBManager', ['''\
>>> from lazyllm.components import MongoDBManager
>>> mgr = MongoDBManager(
...     user="admin",
...     password="123456",
...     host="localhost",
...     port=27017,
...     db_name="mydb",
...     collection_name="books"
... )
>>> result = mgr.execute_query('[{"$match": {"author": "Tolstoy"}}]')
>>> print(result)
... '[{"title": "War and Peace", "author": "Tolstoy"}]'
'''])


add_chinese_doc(
    "MongoDBManager.get_client",
    """\
这是一个上下文管理器，它创建并返回一个数据库会话连接对象，并在使用完成后自动关闭会话。
使用方式例如：
with mongodb_manager.get_client() as client:
    all_dbs = client.list_database_names()

**Returns:**\n
- pymongo.MongoClient: 连接 MongoDB 数据库的对象
""",
)

add_english_doc(
    "MongoDBManager.get_client",
    """\
This is a context manager that creates a database session, yields it for use, and closes the session when done.
Usage example:
with mongodb_manager.get_client() as client:
    all_dbs = client.list_database_names()

**Returns:**\n
- pymongo.MongoClient: MongoDB client used to connect to MongoDB database
""",
)

add_chinese_doc(
    "MongoDBManager.check_connection",
    """\
检查当前MongoDBManager的连接状态。

**Returns:**\n
- DBResult: DBResult.status 连接成功(True), 连接失败(False)。DBResult.detail 包含失败信息
""",
)

add_english_doc(
    "MongoDBManager.check_connection",
    """\
Check the current connection status of the MongoDBManager.

**Returns:**\n
- DBResult: DBResult.status True if the connection is successful, False if it fails. DBResult.detail contains failure information.
""",
)

add_chinese_doc(
    "MongoDBManager.set_desc",
    """\
对于MongoDBManager搭配LLM使用自然语言查询的文档集设置其必须的关键字描述。注意，查询需要用到的关系字都必须提供，因为MonoDB无法像SQL数据库一样获得表结构信息

Args:
    schema_desc_dict (dict): 文档集的关键字描述
""",
)

add_english_doc(
    "MongoDBManager.set_desc",
    """\
When using MongoDBManager with LLM to query documents in natural language, set descriptions for the necessary keywords. Note that all relevant keywords needed for queries must be provided because MongoDB cannot obtain like structural information like a SQL database.

Args:
    tables_desc_dict (dict): descriptive comment for documents
""",
)

add_chinese_doc(
    "SqlCall",
    """\
SqlCall 是一个扩展自 ModuleBase 的类,提供了使用语言模型(LLM)生成和执行 SQL 查询的接口。
它设计用于与 SQL 数据库交互,从语言模型的响应中提取 SQL 查询,执行这些查询,并返回结果或解释。

Arguments:
    llm: 用于生成和解释 SQL 查询及解释的大语言模型。
    sql_manager (SqlManager): 一个 SqlManager 实例，用于处理与 SQL 数据库的交互。
    sql_examples (str, 可选): JSON字符串表示的自然语言转到SQL语句的示例，格式为[{"Question": "查询表中与smith同部门的人员名字", "Answer": "SELECT...;"}]
    use_llm_for_sql_result (bool, 可选): 默认值为True。如果设置为False, 则只输出JSON格式表示的sql执行结果；True则会使用LLM对sql执行结果进行解读并返回自然语言结果。
    return_trace (bool, 可选): 如果设置为 True,则将结果记录在trace中。默认为 False。
""",
)

add_english_doc(
    "SqlCall",
    """\
SqlCall is a class that extends ModuleBase and provides an interface for generating and executing SQL queries using a language model (LLM).
It is designed to interact with a SQL database, extract SQL queries from LLM responses, execute those queries, and return results or explanations.

Arguments:
    llm: A language model to be used for generating and interpreting SQL queries and explanations.
    sql_manager (SqlManager): An instance of SqlManager that handles interaction with the SQL database.
    sql_examples (str, optional): An example of converting natural language represented by a JSON string into an SQL statement, formatted as: [{"Question": "Find the names of people in the same department as Smith", "Answer": "SELECT...;"}]
    use_llm_for_sql_result (bool, optional): Default is True. If set to False, the module will only output raw SQL results in JSON without further processing.
    return_trace (bool, optional): If set to True, the results will be recorded in the trace. Defaults to False.
""",
)

add_example(
    "SqlCall",
    """\
    >>> # First, run SqlManager example
    >>> import lazyllm
    >>> from lazyllm.tools import SQLManger, SqlCall
    >>> sql_tool = SQLManger("personal.db")
    >>> from lazyllm.tools import SQLManger, SqlCall
    >>> sql_tool = SQLManger("personal.db")
    >>> sql_llm = lazyllm.OnlineChatModule(model="gpt-4o", source="openai", base_url="***")
    >>> sql_call = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=True)
    >>> print(sql_call("去年一整年销售额最多的员工是谁?"))
""",
)

# ---------------------------------------------------------------------------- #

add_chinese_doc("HttpTool", """
用于访问第三方服务和执行自定义代码的模块。参数中的 `params` 和 `headers` 的 value，以及 `body` 中可以包含形如 `{{variable}}` 这样用两个花括号标记的模板变量，然后在调用的时候通过参数来替换模板中的值。参考 [[lazyllm.tools.HttpTool.forward]] 中的使用说明。

Args:
    method (str, optional): 指定 http 请求方法，参考 `https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods`。
    url (str, optional): 要访问的 url。如果该字段为空，则表示该模块不需要访问第三方服务。
    params (Dict[str, str], optional): 请求 url 需要填充的 params 字段。如果 url 为空，该字段会被忽略。
    headers (Dict[str, str], optional): 访问 url 需要填充的 header 字段。如果 url 为空，该字段会被忽略。
    body (Dict[str, str], optional): 请求 url 需要填充的 body 字段。如果 url 为空，该字段会被忽略。
    timeout (int): 请求超时时间，单位是秒，默认值是 10。
    proxies (Dict[str, str], optional): 指定请求 url 时所使用的代理。代理格式参考 `https://www.python-httpx.org/advanced/proxies`。
    code_str (str, optional): 一个字符串，包含用户定义的函数。如果参数 `url` 为空，则直接执行该函数，执行时所有的参数都会转发给该函数；如果 `url` 不为空，该函数的参数为请求 url 返回的结果，此时该函数作为 url 返回后的后处理函数。
    vars_for_code (Dict[str, Any]): 一个字典，传入运行 code 所需的依赖及变量。
    outputs (Optional[List[str]]): 期望提取的输出字段名。
    extract_from_result (Optional[bool]): 是否从响应字典中直接提取指定字段。
    outputs (Optional[List[str]]): 期望提取的输出字段名。
    extract_from_result (Optional[bool]): 是否从响应字典中直接提取指定字段。
""")

add_english_doc("HttpTool", """
Module for accessing third-party services and executing custom code. The values in `params` and `headers`, as well as in body, can include template variables marked with double curly braces like `{{variable}}`, which are then replaced with actual values through parameters when called. Refer to the usage instructions in [[lazyllm.tools.HttpTool.forward]].

Args:
    method (str, optional): Specifies the HTTP request method, refer to `https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods`.
    url (str, optional): The URL to access. If this field is empty, it indicates that the module does not need to access third-party services.
    params (Dict[str, str], optional): Params fields to be filled when requesting the URL. If the URL is empty, this field will be ignored.
    headers (Dict[str, str], optional): Header fields to be filled when accessing the URL. If the URL is empty, this field will be ignored.
    body (Dict[str, str], optional): Body fields to be filled when requesting the URL. If the URL is empty, this field will be ignored.
    timeout (int): Request timeout in seconds, default value is 10.
    proxies (Dict[str, str], optional): Specifies the proxies to be used when requesting the URL. Proxy format refer to `https://www.python-httpx.org/advanced/proxies`.
    code_str (str, optional): A string containing a user-defined function. If the parameter url is empty, execute this function directly, forwarding all arguments to it; if url is not empty, the parameters of this function are the results returned from the URL request, and in this case, the function serves as a post-processing function for the URL response.
    vars_for_code (Dict[str, Any]): A dictionary that includes dependencies and variables required for running the code.
    outputs (Optional[List[str]]): Names of expected output fields.
    extract_from_result (Optional[bool]): Whether to extract fields directly from response dict using `outputs`.
    outputs (Optional[List[str]]): Names of expected output fields.
    extract_from_result (Optional[bool]): Whether to extract fields directly from response dict using `outputs`.
""")

add_example("HttpTool", """
from lazyllm.tools import HttpTool

code_str = "def identity(content): return content"
tool = HttpTool(method='GET', url='http://www.sensetime.com/', code_str=code_str)
ret = tool()
""")

add_chinese_doc("HttpTool.forward", """
用于执行初始化时指定的操作：请求指定的 url 或者执行传入的函数。一般不直接调用，而是通过基类的 `__call__` 来调用。如果构造函数的 `url` 参数不为空，则传入的所有参数都会作为变量，用于替换在构造函数中使用 `{{}}` 标记的模板参数；如果构造函数的参数 `url` 为空，并且 `code_str` 不为空，则传入的所有参数都会作为 `code_str` 中所定义函数的参数。
""")

add_english_doc("HttpTool.forward", """
Used to perform operations specified during initialization: request the specified URL or execute the passed function. Generally not called directly, but through the base class's `__call__`. If the `url` parameter in the constructor is not empty, all passed parameters will be used as variables to replace template parameters marked with `{{}}` in the constructor; if the `url` parameter in the constructor is empty and `code_str` is not empty, all passed parameters will be used as arguments for the function defined in `code_str`.
""")

add_example("HttpTool.forward", """
from lazyllm.tools import HttpTool

code_str = "def exp(v, n): return v ** n"
tool = HttpTool(code_str=code_str)
assert tool(v=10, n=2) == 100
""")

add_tools_chinese_doc("Weather", """
创建用于查询天气的工具。
""")

add_tools_english_doc("Weather", """
Create a tool for querying weather.
""")

add_tools_example("Weather", """
from lazyllm.tools.tools import Weather

weather = Weather()
""")

add_tools_chinese_doc("Weather.forward", """
查询某个城市的天气。接收的城市输入最小范围为地级市，如果是直辖市则最小范围为区。输入的城市或区名称不带后缀的“市”或者“区”。参考下面的例子。

Args:
    city_name (str): 需要获取天气的城市名称。
""")

add_tools_english_doc("Weather.forward", """
Query the weather of a specific city. The minimum input scope for cities is at the prefecture level, and for municipalities, it is at the district level. The input city or district name should not include the suffix "市" (city) or "区" (district). Refer to the examples below.

Args:
    city_name (str): The name of the city for which weather information is needed.
""")

add_tools_example("Weather.forward", """
from lazyllm.tools.tools import Weather

weather = Weather()
res = weather('海淀')
""")

add_tools_chinese_doc("GoogleSearch", """
通过 Google 搜索指定的关键词。

Args:
    custom_search_api_key (str): 用户申请的 Google API key。
    search_engine_id (str): 用户创建的用于检索的搜索引擎 id。
    timeout (int): 搜索请求的超时时间，单位是秒，默认是 10。
    proxies (Dict[str, str], optional): 请求时所用的代理服务。格式参考 `https://www.python-httpx.org/advanced/proxies`。
""")

add_tools_english_doc("GoogleSearch", """
Search for specified keywords through Google.

Args:
    custom_search_api_key (str): The Google API key applied by the user.
    search_engine_id (str): The ID of the search engine created by the user for retrieval.
    timeout (int): The timeout for the search request, in seconds, default is 10.
    proxies (Dict[str, str], optional): The proxy services used during the request. Format reference `https://www.python-httpx.org/advanced/proxies`.
""")

add_tools_example("GoogleSearch", """
from lazyllm.tools.tools import GoogleSearch

key = '<your_google_search_api_key>'
cx = '<your_search_engine_id>'

google = GoogleSearch(custom_search_api_key=key, search_engine_id=cx)
""")

add_tools_chinese_doc("GoogleSearch.forward", """
执行搜索请求。

Args:
    query (str): 要检索的关键词。
    date_restrict (str): 要检索内容的时效性。默认检索一个月内的网页（`m1`）。参数格式可以参考 `https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn`。
    search_engine_id (str, optional): 用于检索的搜索引擎 id。如果该值为空，则使用构造函数中传入的值。
""")

add_tools_english_doc("GoogleSearch.forward", """
Execute search request.

Args:
    query (str): Keywords to retrieve.
    date_restrict (str): Timeliness of the content to retrieve. Defaults to web pages within one month (m1). Refer to `https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn` for parameter format.
    search_engine_id (str, optional): Search engine ID for retrieval. If this value is empty, the value passed in the constructor is used.
""")

add_tools_example("GoogleSearch.forward", """
from lazyllm.tools.tools import GoogleSearch

key = '<your_google_search_api_key>'
cx = '<your_search_engine_id>'

google = GoogleSearch(key, cx)
res = google(query='商汤科技', date_restrict='m1')
""")

add_tools_chinese_doc('Calculator', '''
这是一个计算器应用，可以计算用户输入的表达式的值。
''')

add_tools_english_doc('Calculator', '''
This is a calculator application that can calculate the value of expressions entered by the user.
''')

add_tools_example('Calculator', '''
from lazyllm.tools.tools import Calculator
calc = Calculator()
''')

add_tools_chinese_doc('Calculator.forward', '''
计算用户输入的表达式的值。

Args:
    exp (str): 需要计算的表达式的值。必须符合 Python 计算表达式的语法。可使用 Python math 库中的数学函数。
''')

add_tools_english_doc('Calculator.forward', '''
Calculate the value of the user input expression.

Args:
    exp (str): The expression to be calculated. It must conform to the syntax for evaluating expressions in Python. Mathematical functions from the Python math library can be used.
''')

add_tools_example('Calculator.forward', '''
from lazyllm.tools.tools import Calculator
calc = Calculator()
''')

add_tools_chinese_doc('TencentSearch', '''
这是一个搜索增强工具。
''')

add_tools_english_doc('TencentSearch', '''
This is a search enhancement tool.
''')

add_tools_example('TencentSearch', '''
from lazyllm.tools.tools import TencentSearch
secret_id = '<your_secret_id>'
secret_key = '<your_secret_key>'
searcher = TencentSearch(secret_id, secret_key)
''')

add_tools_chinese_doc('TencentSearch.forward', '''
搜索用户输入的查询。

Args:
    query (str): 用户待查询的内容。
''')

add_tools_english_doc('TencentSearch.forward', '''
Searches for the query entered by the user.

Args:
    query (str): The content that the user wants to query.
''')

add_tools_example('TencentSearch.forward', '''
from lazyllm.tools.tools import TencentSearch
secret_id = '<your_secret_id>'
secret_key = '<your_secret_key>'
searcher = TencentSearch(secret_id, secret_key)
res = searcher('calculus')
''')


# ---------------------------------------------------------------------------- #

# mcp/client.py

add_english_doc('MCPClient', '''\
MCP client that can be used to connect to an MCP server. It supports both local servers (through stdio client) and remote servers (through sse client).

If the 'command_or_url' is a url string (started with 'http' or 'https'), a remote server will be connected, otherwise a local server will be started and connected.

Args:
    command_or_url (str): The command or url string, which will be used to start a local server or connect to a remote server.
    args (list[str], optional): Arguments list used for starting a local server, if you want to connect to a remote server, this argument is not needed. (default is [])
    env (dict[str, str], optional): Environment variables dictionary used in tools, for example some api keys. (default is None)
    headers(dict[str, Any], optional): HTTP headers used in sse client connection. (default is None)
    timeout (float, optional): Timeout for sse client connection, in seconds. (default is 5)
''')

add_chinese_doc('MCPClient', '''\
MCP客户端，用于连接MCP服务器。同时支持本地服务器和sse服务器。
MCP客户端，用于连接MCP服务器。同时支持本地服务器和sse服务器。



Args:
    command_or_url (str): 用于启动本地服务器或连接远程服务器的命令或 URL 字符串。
    args (list[str], optional): 用于启动本地服务器的参数列表；如果要连接远程服务器，则无需此参数。（默认值为[]）
    env (dict[str, str], optional): 工具中使用的环境变量，例如一些 API 密钥。（默认值为None）
    headers(dict[str, Any], optional): 用于sse客户端连接的HTTP头。（默认值为None）
    timeout (float, optional): sse客户端连接的超时时间，单位为秒。(默认值为5)
''')


add_english_doc('MCPClient.call_tool', '''\
Calls one of the tools provided in the toolset of the connected MCP server via the MCP client and returns the result.

Args:
    tool_name (str): The name of the tool.
    arguments (dict): The parameters for the tool.
''')

add_chinese_doc('MCPClient.call_tool', '''\
通过MCP客户端调用连接的MCP服务器提供的工具集中的某一个工具，并返回结果。

Args:
    tool_name (str): 工具名称。
    arguments (dict): 工具传参。
''')


add_english_doc('MCPClient.list_tools', '''\
Retrieves the list of tools from the currently connected MCP client.
''')

add_chinese_doc('MCPClient.list_tools', '''\
获取当前连接MCP客户端的工具列表。
''')


add_english_doc('MCPClient.aget_tools', '''\
Used to convert the tool set from the MCP server into a list of functions available for LazyLLM and return them.

The allowed_tools parameter is used to specify the list of tools to be returned. If None, all tools will be returned.

Args: 
    allowed_tools (list[str], optional): The list of tools expected to be returned. Defaults to None, meaning that all tools will be returned.
''')

add_chinese_doc('MCPClient.aget_tools', '''\
用于将MCP服务器中的工具集转换为LazyLLM可用的函数列表，并返回。

allowed_tools参数用于指定要返回的工具列表，默认为None，表示返回所有工具。

Args:
    allowed_tools (list[str], optional): 期望返回的工具列表，默认为None，表示返回所有工具。
''')


add_example('MCPClient', '''\
>>> from lazyllm.tools import MCPClient
>>> mcp_server_configs = {
...     "filesystem": {
...         "command": "npx",
...         "args": [
...             "-y",
...             "@modelcontextprotocol/server-filesystem",
...             "./",
...         ]
...     }
... }
>>> file_sys_config = mcp_server_configs["filesystem"]
>>> file_client = MCPClient(
...     command_or_url=file_sys_config["command"],
...     args=file_sys_config["args"],
... )
>>> from lazyllm import OnlineChatModule
>>> from lazyllm.tools.agent.reactAgent import ReactAgent
>>> llm=OnlineChatModule(source="deepseek", stream=False)
>>> agent = ReactAgent(llm.share(), file_client.get_tools())
>>> print(agent("Write a Chinese poem about the moon, and save it to a file named 'moon.txt".))
''')


# ---------------------------------------------------------------------------- #

# mcp/tool_adaptor.py

add_english_doc('mcp.tool_adaptor.generate_lazyllm_tool', '''\
Dynamically build a function for the LazyLLM agent based on a tool provided by the MCP server.

Args:
    client (mcp.ClientSession): MCP client which connects to the MCP server.
    mcp_tool (mcp.types.Tool): A tool provided by the MCP server.
''')

add_chinese_doc('mcp.tool_adaptor.generate_lazyllm_tool', '''\
将 MCP 服务器提供的工具转换为 LazyLLM 代理使用的函数。

Args:
    client (mcp.ClientSession): 连接到MCP服务器的MCP客户端。
    mcp_tool (mcp.types.Tool): 由MCP服务器提供的工具。
''')

add_english_doc('rag.utils.DocListManager.table_inited', '''\
Checks if the database table `documents` is initialized. This method ensures thread-safety when accessing the database.
`table_inited(self)`
Determines whether the `documents` table exists in the database.
Returns:
    bool: `True` if the `documents` table exists, `False` otherwise.
Notes:
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe access to the database.
    - Establishes a connection to the SQLite database at `self._db_path` with the `check_same_thread` option.
    - Executes the SQL query: `SELECT name FROM sqlite_master WHERE type='table' AND name='documents'` to check for the table.
''')

add_chinese_doc('rag.utils.DocListManager.table_inited', '''\
检查数据库中的 `documents` 表是否已初始化。此方法在访问数据库时确保线程安全。
`table_inited(self)`
判断数据库中是否存在 `documents` 表。
返回值:
    bool: 如果 `documents` 表存在，返回 `True`；否则返回 `False`。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保对数据库的安全访问。
    - 通过 `self._db_path` 连接 SQLite 数据库，并使用 `check_same_thread` 配置选项。
    - 执行 SQL 查询：`SELECT name FROM sqlite_master WHERE type='table' AND name='documents'` 来检查表是否存在。
''')

add_english_doc('rag.utils.DocListManager.validate_paths', '''\
Validates a list of file paths to ensure they are ready for processing.
`validate_paths(self, paths: List[str]) -> Tuple[bool, str, List[bool]]`
This method checks whether the provided paths are new, already processed, or currently being processed. It ensures there are no conflicts in processing the documents.
Args
    paths (List[str]): A list of file paths to validate.
Returns:
    Tuple[bool, str, List[bool]]: A tuple containing:
        - `bool`: `True` if all paths are valid, `False` otherwise.
        - `str`: A message indicating success or the reason for failure.
        - `List[bool]`: A list where each element corresponds to whether a path is new (`True`) or already exists (`False`).
Notes:
    - If any document is still being processed or needs reparsing, the method returns `False` with an appropriate error message.
    - The method uses a database session and thread-safe lock (`self._db_lock`) to retrieve document status information.
    - Unsafe statuses include `working` and `waiting`.
''')

add_chinese_doc('rag.utils.DocListManager.validate_paths', '''\
验证一组文件路径，以确保它们可以被正常处理。
`validate_paths(self, paths: List[str]) -> Tuple[bool, str, List[bool]]`
此方法检查提供的路径是否是新的、已处理的或当前正在处理的，并确保处理文档时不会发生冲突。
参数:
    paths (List[str]): 要验证的文件路径列表。
返回值:
    Tuple[bool, str, List[bool]]: 返回一个元组，包括：
        - `bool`: 如果所有路径有效，则返回 `True`；否则返回 `False`。
        - `str`: 表示成功或失败原因的消息。
        - `List[bool]`: 一个布尔值列表，每个元素对应一个路径是否为新路径（`True` 表示新路径，`False` 表示已存在）。
说明:
    - 如果任何文档仍在处理中或需要重新解析，该方法会返回 `False`，并附带相应的错误消息。
    - 方法通过数据库会话和线程安全锁 (`self._db_lock`) 检索文档状态信息。
    - 不安全状态包括 `working` 和 `waiting`。
''')

add_english_doc('rag.utils.DocListManager.update_need_reparsing', '''\
Updates the `need_reparse` status of a document in the `KBGroupDocuments` table.
`update_need_reparsing(self, doc_id: str, need_reparse: bool, group_name: Optional[str] = None)`
This method sets the `need_reparse` flag for a specific document, optionally scoped to a given group.
Args:
    doc_id (str): The ID of the document to update.
    need_reparse (bool): The new value for the `need_reparse` flag.
    group_name (Optional[str]): If provided, the update will be applied only to the specified group.
Notes:
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The `group_name` parameter allows scoping the update to a specific group; if not provided, the update applies to all groups containing the document.
    - The method commits the change to the database immediately.
''')

add_chinese_doc('rag.utils.DocListManager.update_need_reparsing', '''\
更新 `KBGroupDocuments` 表中某个文档的 `need_reparse` 状态。
`update_need_reparsing(self, doc_id: str, need_reparse: bool, group_name: Optional[str] = None)`
此方法设置指定文档的 `need_reparse` 标志，并可选限定到特定分组。
参数:
    doc_id (str): 要更新的文档ID。
    need_reparse (bool): `need_reparse` 标志的新值。
    group_name (Optional[str]): 如果提供，仅对指定分组应用更新；如果未提供，则对包含该文档的所有分组应用更新。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - `group_name` 参数允许将更新限定到特定分组；如果未提供，则更新应用于包含该文档的所有分组。
    - 方法会立刻将更改提交到数据库。
''')

add_english_doc('rag.utils.DocListManager.list_files', '''\
Lists files from the `documents` table with optional filtering, limiting, and returning details.
`list_files(self, limit: Optional[int] = None, details: bool = False, status: Union[str, List[str]] = DocListManager.Status.all, exclude_status: Optional[Union[str, List[str]]] = None)`
This method retrieves file IDs or detailed file information from the database, based on the specified filtering conditions.
Args:
    limit (Optional[int]): Maximum number of files to return. If `None`, all matching files will be returned.
    details (bool): Whether to return detailed file information (`True`) or just file IDs (`False`).
    status (Union[str, List[str]]): The status or list of statuses to include in the results. Defaults to all statuses.
    exclude_status (Optional[Union[str, List[str]]]): The status or list of statuses to exclude from the results. Defaults to `None`.
Returns:
    List: A list of file IDs if `details=False`, or a list of detailed file rows if `details=True`.
Notes:
    - The method constructs a query dynamically based on the provided `status` and `exclude_status` conditions.
    - A thread-safe lock (`self._db_lock`) ensures safe database access.
    - The `LIMIT` clause is applied if `limit` is specified.
''')

add_chinese_doc('rag.utils.DocListManager.list_files', '''\
从 `documents` 表中列出文件，并支持过滤、限制返回结果以及返回详细信息。
`list_files(self, limit: Optional[int] = None, details: bool = False, status: Union[str, List[str]] = DocListManager.Status.all, exclude_status: Optional[Union[str, List[str]]] = None)`
此方法根据指定的条件，从数据库中检索文件ID或详细文件信息。
参数:
    limit (Optional[int]): 返回的最大文件数量。如果为 `None`，则返回所有匹配的文件。
    details (bool): 是否返回详细的文件信息（`True`）或仅返回文件ID（`False`）。
    status (Union[str, List[str]]): 要包含的状态或状态列表，默认为所有状态。
    exclude_status (Optional[Union[str, List[str]]]): 要排除的状态或状态列表，默认为 `None`。
返回值:
    List: 如果 `details=False`，则返回文件ID列表；如果 `details=True`，则返回详细文件行的列表。
说明:
    - 该方法根据 `status` 和 `exclude_status` 条件动态构造查询。
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - 如果指定了 `limit`，查询会附加 `LIMIT` 子句。
''')

add_english_doc('rag.utils.DocListManager.get_docs', '''\
Fetch documents from the database based on a list of document IDs.
`get_docs(self, doc_ids: List[str]) -> List[KBDocument]`
This method retrieves document objects of type `KBDocument` from the database for the provided list of document IDs.
Args:
    doc_ids (List[str]): A list of document IDs to fetch.
Returns:
    List[KBDocument]: A list of `KBDocument` objects corresponding to the provided document IDs. If no documents are found, an empty list is returned.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The query filters documents using the `doc_id` field with an SQL `IN` clause.
    - If `doc_ids` is empty, the function will return an empty list without querying the database.
''')

add_chinese_doc('rag.utils.DocListManager.get_docs', '''\
根据文档 ID 列表从数据库中获取文档对象。
`get_docs(self, doc_ids: List[str]) -> List[KBDocument]`
此方法从数据库中检索类型为 `KBDocument` 的文档对象，基于提供的文档 ID 列表。
参数:
    doc_ids (List[str]): 要获取的文档 ID 列表。
返回值:
    List[KBDocument]: 与提供的文档 ID 对应的 `KBDocument` 对象列表。如果没有找到文档，将返回空列表。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问的安全性。
    - 查询使用 SQL 的 `IN` 子句，通过 `doc_id` 字段进行过滤。
    - 如果 `doc_ids` 为空，函数将直接返回空列表，而不会查询数据库。
''')

add_english_doc('rag.utils.DocListManager.fetch_docs_changed_meta', '''\
Fetch documents with changed metadata for a specific group and reset their `new_meta` field to `None`.
`fetch_docs_changed_meta(self, group: str) -> List[DocMetaChangedRow]`
This method retrieves all documents where metadata has changed (`new_meta` is not `None`) for the given group. After fetching, it resets the `new_meta` field to `None` for those documents.
Args:
    group (str): The name of the group to filter documents by.
Returns:
    List[DocMetaChangedRow]: A list of rows, where each row contains the `doc_id` and the `new_meta` field of documents with changed metadata.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - It performs a SQL join between `KBDocument` and `KBGroupDocuments` to retrieve the relevant rows.
    - After fetching, it updates the `new_meta` field of the affected rows to `None` and commits the changes to the database.
''')

add_chinese_doc('rag.utils.DocListManager.fetch_docs_changed_meta', '''\
获取指定组中元数据已更改的文档，并将其 `new_meta` 字段重置为 `None`。
`fetch_docs_changed_meta(self, group: str) -> List[DocMetaChangedRow]`
此方法检索元数据已更改（即 `new_meta` 不为 `None`）的所有文档，基于提供的组名。检索后，会将这些文档的 `new_meta` 字段重置为 `None`。
参数:
    group (str): 用于过滤文档的组名。
返回值:
    List[DocMetaChangedRow]: 包含文档 `doc_id` 和 `new_meta` 字段的行列表，表示元数据已更改的文档。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - 方法通过 SQL `JOIN` 操作连接 `KBDocument` 和 `KBGroupDocuments` 表以检索相关行。
    - 在获取数据后，将受影响行的 `new_meta` 字段更新为 `None`，并将更改提交到数据库。
''')

add_english_doc('rag.utils.DocListManager.list_kb_group_files', '''\
List files in a specific knowledge base (KB) group with optional filters, limiting, and details.
`list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False, status: Union[str, List[str]] = DocListManager.Status.all, exclude_status: Optional[Union[str, List[str]]] = None, upload_status: Union[str, List[str]] = DocListManager.Status.all, exclude_upload_status: Optional[Union[str, List[str]]] = None, need_reparse: Optional[bool] = None)`
This method retrieves files from the `kb_group_documents` table, optionally filtering by group, document status, upload status, and whether reparsing is needed.
Args:
    group (str): The name of the KB group to filter files by. Defaults to `None` (no group filter).
    limit (Optional[int]): Maximum number of files to return. If `None`, returns all matching files.
    details (bool): Whether to return detailed file information (`True`) or only file IDs and paths (`False`).
    status (Union[str, List[str]]): The KB group status or list of statuses to include in the results. Defaults to all statuses.
    exclude_status (Optional[Union[str, List[str]]): The KB group status or list of statuses to exclude from the results. Defaults to `None`.
    upload_status (Union[str, List[str]]): The document upload status or list of statuses to include in the results. Defaults to all statuses.
    exclude_upload_status (Optional[Union[str, List[str]]): The document upload status or list of statuses to exclude from the results. Defaults to `None`.
    need_reparse (Optional[bool]): Whether to filter files that need reparsing (`True`) or not (`False`). Defaults to `None` (no filtering).
Returns:
    List: If `details=False`, returns a list of tuples containing `(doc_id, path)`. 
          If `details=True`, returns a list of detailed rows with additional metadata.
Notes:
    - This method constructs a SQL query dynamically based on the provided filters.
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - If `status` or `upload_status` are provided as lists, they are processed with SQL `IN` clauses.
''')

add_chinese_doc('rag.utils.DocListManager.list_kb_group_files', '''\
列出指定知识库 (KB) 组中的文件，并支持过滤、限制返回以及返回详细信息。
`list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False, status: Union[str, List[str]] = DocListManager.Status.all, exclude_status: Optional[Union[str, List[str]]] = None, upload_status: Union[str, List[str]] = DocListManager.Status.all, exclude_upload_status: Optional[Union[str, List[str]]] = None, need_reparse: Optional[bool] = None)`
此方法从 `kb_group_documents` 表中检索文件，支持基于组名、文档状态、上传状态以及是否需要重新解析的过滤。
参数:
    group (str): 用于过滤文件的 KB 组名。默认为 `None`（不过滤组名）。
    limit (Optional[int]): 返回的最大文件数量。如果为 `None`，则返回所有匹配的文件。
    details (bool): 是否返回详细的文件信息（`True`）或仅返回文件 ID 和路径（`False`）。
    status (Union[str, List[str]]): 要包含在结果中的 KB 组状态或状态列表。默认为所有状态。
    exclude_status (Optional[Union[str, List[str]]): 要从结果中排除的 KB 组状态或状态列表。默认为 `None`。
    upload_status (Union[str, List[str]]): 要包含在结果中的文档上传状态或状态列表。默认为所有状态。
    exclude_upload_status (Optional[Union[str, List[str]]): 要从结果中排除的文档上传状态或状态列表。默认为 `None`。
    need_reparse (Optional[bool]): 是否过滤需要重新解析的文件（`True`）或不需要重新解析的文件（`False`）。默认为 `None`（不进行过滤）。
返回值:
    List: 如果 `details=False`，返回包含 `(doc_id, path)` 的元组列表。
          如果 `details=True`，返回包含附加元数据的详细行列表。
说明:
    - 方法根据提供的过滤条件动态构建 SQL 查询。
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - 如果 `status` 或 `upload_status` 参数为列表，则会使用 SQL 的 `IN` 子句进行处理。
''')

add_english_doc('rag.utils.DocListManager.add_files', '''\
Add multiple files to the document list with optional metadata, status, and batch processing.
`add_files(self, files: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, status: Optional[str] = Status.waiting, batch_size: int = 64) -> List[DocPartRow]`
This method adds a list of files to the database with optional metadata and a specified initial status. Files are processed in batches for efficiency. After adding the documents, they are associated with the default knowledge base (KB) group.
Args:
    files (List[str]): A list of file paths to add to the database.
    metadatas (Optional[List[Dict[str, Any]]]): A list of metadata dictionaries corresponding to the files. If `None`, no metadata will be associated. Defaults to `None`.
    status (Optional[str]): The initial status for the added files. Defaults to `Status.waiting`.
    batch_size (int): The number of files to process in each batch. Defaults to 64.
Returns:
    List[DocPartRow]: A list of `DocPartRow` objects representing the added files and their associated information.
Notes:
    - The method first creates document records using the `_add_doc_records` helper function.
    - After the files are added, they are automatically linked to the default KB group (`DocListManager.DEFAULT_GROUP_NAME`).
    - Batch processing ensures scalability when adding a large number of files.
''')

add_chinese_doc('rag.utils.DocListManager.add_files', '''\
批量向文档列表中添加文件，可选附加元数据、状态，并支持分批处理。
`add_files(self, files: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, status: Optional[str] = Status.waiting, batch_size: int = 64) -> List[DocPartRow]`
此方法将文件列表添加到数据库中，并为每个文件设置可选的元数据和初始状态。文件会以批量方式处理以提高效率。在文件添加完成后，它们会自动关联到默认的知识库 (KB) 组。
参数:
    files (List[str]): 要添加到数据库的文件路径列表。
    metadatas (Optional[List[Dict[str, Any]]]): 与文件对应的元数据字典列表。如果为 `None`，则不会附加元数据。默认为 `None`。
    status (Optional[str]): 添加文件的初始状态。默认为 `Status.waiting`。
    batch_size (int): 每批处理的文件数量。默认为 64。
返回值:
    List[DocPartRow]: 包含已添加文件及其相关信息的 `DocPartRow` 对象列表。
说明:
    - 方法首先通过辅助函数 `_add_doc_records` 创建文档记录。
    - 文件添加后，会自动关联到默认的知识库组 (`DocListManager.DEFAULT_GROUP_NAME`)。
    - 批量处理确保在添加大量文件时具有良好的可扩展性。
''')


#delete_unreferenced_doc
add_english_doc('rag.utils.DocListManager.delete_unreferenced_doc', '''\
Delete documents marked as "deleting" and no longer referenced in the database.
`delete_unreferenced_doc(self)`
This method removes documents from the database that meet the following conditions:
1. Their status is set to `DocListManager.Status.deleting`.
2. Their reference count (`count`) is 0.
''')

add_chinese_doc('rag.utils.DocListManager.delete_unreferenced_doc', '''\
删除数据库中标记为 "删除中" 且不再被引用的文档。
`delete_unreferenced_doc(self)`
此方法从数据库中删除满足以下条件的文档：
1. 文档状态为 `DocListManager.Status.deleting`。
2. 文档的引用计数 (`count`) 为 0。
''')

#get_docs_need_reparse
add_english_doc('rag.utils.DocListManager.get_docs_need_reparse', '''\
Retrieve documents that require reparsing for a specific group.
`get_docs_need_reparse(self, group: str) -> List[KBDocument]`
This method fetches documents that are marked as needing reparsing (`need_reparse=True`) for the given group. Only documents with a status of `success` or `failed` are included in the results.
Args:
    group (str): The name of the group to filter documents by.
Returns:
    List[KBDocument]: A list of `KBDocument` objects that need reparsing.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The query performs a SQL `JOIN` between `KBDocument` and `KBGroupDocuments` to filter by group and reparse status.
    - Documents with `need_reparse=True` and a status of `success` or `failed` are considered for reparsing.
''')

add_chinese_doc('rag.utils.DocListManager.get_docs_need_reparse', '''\
获取需要重新解析的指定组中的文档。
`get_docs_need_reparse(self, group: str) -> List[KBDocument]`
此方法检索标记为需要重新解析 (`need_reparse=True`) 的文档，基于提供的组名。仅包含状态为 `success` 或 `failed` 的文档。
参数:
    group (str): 用于过滤文档的组名。
返回值:
    List[KBDocument]: 需要重新解析的 `KBDocument` 对象列表。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - 查询通过 SQL `JOIN` 操作连接 `KBDocument` 和 `KBGroupDocuments` 表，并基于组名和重新解析状态进行过滤。
    - 仅状态为 `success` 或 `failed` 且 `need_reparse=True` 的文档会被检索出来。
''')

add_english_doc('rag.utils.DocListManager.get_existing_paths_by_pattern', '''\
Retrieve existing document paths that match a given pattern.
`get_existing_paths_by_pattern(self, pattern: str) -> List[str]`
This method fetches all document paths from the database that match the provided SQL `LIKE` pattern.
Args:
    pattern (str): The SQL `LIKE` pattern to filter document paths. For example, `%example%` matches paths containing the word "example".
Returns:
    List[str]: A list of document paths that match the given pattern. If no paths match, an empty list is returned.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The `LIKE` operator in the SQL query is used to perform pattern matching on document paths.

''')

add_chinese_doc('rag.utils.DocListManager.get_existing_paths_by_pattern', '''\
根据给定的模式，检索符合条件的文档路径。
`get_existing_paths_by_pattern(self, pattern: str) -> List[str]`
此方法从数据库中获取所有符合提供的 SQL `LIKE` 模式的文档路径。
参数:
    pattern (str): 用于过滤文档路径的 SQL `LIKE` 模式。例如，`%example%` 匹配包含单词 "example" 的路径。
返回值:
    List[str]: 符合给定模式的文档路径列表。如果没有匹配的路径，则返回空列表。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - SQL 查询中的 `LIKE` 操作符用于对文档路径进行模式匹配。
''')

add_english_doc('rag.utils.DocListManager.enable_path_monitoring', '''\
Enable or disable path monitoring for the document manager.
`enable_path_monitoring(self, val: bool)`
This method enables or disables the path monitoring functionality in the document manager. When enabled, a monitoring thread starts to handle path-related operations. When disabled, the thread stops and joins (waits for it to terminate).
Args:
    val (bool): A boolean value indicating whether to enable (`True`) or disable (`False`) path monitoring.
Notes:
    - If `val` is `True`, path monitoring is enabled by setting `_monitor_continue` to `True` and starting the `_monitor_thread`.
    - If `val` is `False`, path monitoring is disabled by setting `_monitor_continue` to `False` and joining the `_monitor_thread` if it is running.
    - This method ensures thread-safe operation when managing the monitoring thread.
''')

add_chinese_doc('rag.utils.DocListManager.enable_path_monitoring', '''\
启用或禁用文档管理器的路径监控功能。
`enable_path_monitoring(self, val: bool)`
此方法用于启用或禁用文档管理器的路径监控功能。当启用时，会启动一个监控线程处理与路径相关的操作；当禁用时，会停止该线程并等待它终止。
参数:
    val (bool): 布尔值，指示是否启用 (`True`) 或禁用 (`False`) 路径监控。
说明:
    - 如果 `val` 为 `True`，路径监控功能会通过将 `_monitor_continue` 设置为 `True` 并启动 `_monitor_thread` 来启用。
    - 如果 `val` 为 `False`，路径监控功能会通过将 `_monitor_continue` 设置为 `False` 并等待 `_monitor_thread` 终止来禁用。
    - 方法在管理监控线程时确保线程操作是安全的。
''')

add_english_doc('rag.global_metadata.GlobalMetadataDesc', '''\
A descriptor for global metadata, defining its type, optional element type, default value, and size constraints.
`class GlobalMetadataDesc`
This class is used to describe metadata properties such as type, optional constraints, and default values. It supports scalar and array data types, with specific size limitations for certain types.
Args:
    data_type (int): The type of the metadata as an integer, representing various data types (e.g., VARCHAR, ARRAY, etc.).
    element_type (Optional[int]): The type of individual elements if `data_type` is an array. Defaults to `None`.
    default_value (Optional[Any]): The default value for the metadata. If not provided, the default will be `None`.
    max_size (Optional[int]): The maximum size or length for the metadata. Required if `data_type` is `VARCHAR` or `ARRAY`.
''')

add_chinese_doc('rag.global_metadata.GlobalMetadataDesc', '''\
用于描述全局元数据的说明符，包括其类型、可选的元素类型、默认值和大小限制。
`class GlobalMetadataDesc`
此类用于描述元数据的属性，例如类型、可选约束和默认值。支持标量和数组数据类型，并对某些类型指定特定的大小限制。
Args:
    data_type (int): 元数据的类型，以整数表示，代表不同的数据类型（例如 VARCHAR、ARRAY 等）。
    element_type (Optional[int]): 如果 `data_type` 是数组，则表示数组中每个元素的类型。默认为 `None`。
    default_value (Optional[Any]): 元数据的默认值。如果未提供，默认值为 `None`。
    max_size (Optional[int]): 元数据的最大大小或长度。如果 `data_type` 为 `VARCHAR` 或 `ARRAY`，则此属性为必填项。
''')

add_english_doc('rag.index_base.IndexBase', '''\
An abstract base class for implementing indexing systems that support updating, removing, and querying document nodes.
`class IndexBase(ABC)`
This abstract base class defines the interface for an indexing system. It requires subclasses to implement methods for updating, removing, and querying document nodes.
''')

add_chinese_doc('rag.index_base.IndexBase', '''\
用于实现索引系统的抽象基类，支持更新、删除和查询文档节点。
`class IndexBase(ABC)`
此抽象基类定义了索引系统的接口，要求子类实现更新、删除和查询文档节点的方法。
''')

add_example('rag.index_base.IndexBase', '''\
>>> from mymodule import IndexBase, DocNode
>>> class MyIndex(IndexBase):
...     def __init__(self):
...         self.nodes = []
...     def update(self, nodes):
...         self.nodes.extend(nodes)
...         print(f"Updated nodes: {nodes}")
...     def remove(self, uids, group_name=None):
...         self.nodes = [node for node in self.nodes if node.uid not in uids]
...         print(f"Removed nodes with uids: {uids}")
...     def query(self, *args, **kwargs):
...         print("Querying nodes...")
...         return self.nodes
>>> index = MyIndex()
>>> doc1 = DocNode(uid="1", content="Document 1")
>>> doc2 = DocNode(uid="2", content="Document 2")
>>> index.update([doc1, doc2])
Updated nodes: [DocNode(uid="1", content="Document 1"), DocNode(uid="2", content="Document 2")]
>>> index.query()
Querying nodes...
[DocNode(uid="1", content="Document 1"), DocNode(uid="2", content="Document 2")]
>>> index.remove(["1"])
Removed nodes with uids: ['1']
>>> index.query()
Querying nodes...
[DocNode(uid="2", content="Document 2")]
''')