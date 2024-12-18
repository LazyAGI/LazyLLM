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
    manager (bool, optional): A flag indicating whether to create a user interface for the document module. Defaults to False.
    launcher (optional): An object or function responsible for launching the server module. If not provided, the default asynchronous launcher from `lazyllm.launchers` is used (`sync=False`).
''')

add_chinese_doc('Document', '''\
初始化一个具有可选用户界面的文档模块。

此构造函数初始化一个可以有或没有用户界面的文档模块。如果启用了用户界面，它还会提供一个ui界面来管理文档操作接口，并提供一个用于用户界面交互的网页。

Args:
    dataset_path (str): 数据集目录的路径。此目录应包含要由文档模块管理的文档。
    embed (Optional[Union[Callable, Dict[str, Callable]]]): 用于生成文档 embedding 的对象。如果需要对文本生成多个 embedding，此处需要通过字典的方式指定多个 embedding 模型，key 标识 embedding 对应的名字, value 为对应的 embedding 模型。
    manager (bool, optional): 指示是否为文档模块创建用户界面的标志。默认为 False
    launcher (optional): 负责启动服务器模块的对象或函数。如果未提供，则使用 `lazyllm.launchers` 中的默认异步启动器 (`sync=False`)。
''')

add_example('Document', '''\
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)  # or documents = Document(dataset_path='your_doc_path', embed={"key": m}, manager=False)
>>> m1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
>>> document1 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, manager=False)
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
>>> def processYml(file, extra_info=None):
...     with open(file, 'r') as f:
...         data = f.read()
...     return [DocNode(text=data, metadata=extra_info or {})]
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
...     def _load_data(self, file, extra_info=None, fs=None):
...         try:
...             import yaml
...         except ImportError:
...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
...         with open(file, 'r') as f:
...             data = yaml.safe_load(f)
...         print("Call the class YmlReader.")
...         return [DocNode(text=data, metadata=extra_info or {})]
... 
>>> def processYml(file, extra_info=None):
...     with open(file, 'r') as f:
...         data = f.read()
...     print("Call the function processYml.")
...     return [DocNode(text=data, metadata=extra_info or {})]
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
The base class of file readers, which inherits from the ModuleBase base class and has Callable capabilities. Subclasses that inherit from this class only need to implement the _load_data function, and its return parameter type is List[DocNode]. Generally, the input parameters of the _load_data function are file (Path), extra_info(Dict), and fs (AbstractFileSystem).

Args:
    args (Any): Pass the corresponding position parameters as needed.
    return_trace (bool): Set whether to record trace logs.
    kwargs (Dict): Pass the corresponding keyword arguments as needed.
''')

add_chinese_doc('rag.readers.ReaderBase', '''
文件读取器的基类，它继承自 ModuleBase 基类，具有 Callable 的能力，继承自该类的子类只需要实现 _load_data 函数即可，它的返回参数类型为 List[DocNode]. 一般 _load_data 函数的入参为 file (Path), extra_info (Dict), fs(AbstractFileSystem) 三个参数。

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
...     def _load_data(self, file: Path, extra_info: Optional[Dict] = None, fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
...         try:
...             import yaml
...         except ImportError:
...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
...         with open(file, 'r') as f:
...             data = yaml.safe_load(f)
...         print("Call the class YmlReader.")
...         return [DocNode(text=data, metadata=extra_info or {})]
... 
>>> files = ["your_yml_files"]
>>> doc = Document(dataset_path="your_files_path", create_ui=False)
>>> reader = doc._impl._reader.load_data(input_files=files)
# Call the class YmlReader.
''')

# ---------------------------------------------------------------------------- #

# rag/rerank.py

add_english_doc('Reranker', '''\
Initializes a Rerank module for postprocessing and reranking of nodes (documents).
This constructor initializes a Reranker module that configures a reranking process based on a specified reranking type. It allows for the dynamic selection and instantiation of reranking kernels (algorithms) based on the type and provided keyword arguments.

Args:
    name: The type of reranker to be used for the postprocessing and reranking process. Defaults to 'Reranker'.
    kwargs: Additional keyword arguments that are passed to the reranker upon its instantiation.

**Detailed explanation of reranker types**

- Reranker: This registered reranking function instantiates a SentenceTransformerRerank reranker with a specified model and top_n parameter. It is designed to rerank nodes based on sentence transformer embeddings.\n
- KeywordFilter: This registered reranking function instantiates a KeywordNodePostprocessor with specified required and excluded keywords. It filters nodes based on the presence or absence of these keywords.
''')

add_chinese_doc('Reranker', '''\
用于创建节点（文档）后处理和重排序的模块。

Args:
    name: 用于后处理和重排序过程的排序器类型。默认为 'Reranker'。
    kwargs: 传递给重新排序器实例化的其他关键字参数。

详细解释排序器类型

  - Reranker: 实例化一个具有指定模型和 top_n 参数的 SentenceTransformerRerank 重排序器。
  - KeywordFilter: 实例化一个具有指定必需和排除关键字的 KeywordNodePostprocessor。它根据这些关键字的存在或缺失来过滤节点。
''')

add_example('Reranker', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, Reranker, Retriever
>>> m = lazyllm.OnlineEmbeddingModule()
>>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> retriever = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
>>> reranker = Reranker(name='ModuleReranker', model='bge-reranker-large', topk=1)
>>> ppl = lazyllm.ActionModule(retriever, reranker)
>>> ppl.start()
>>> print(ppl("user query"))
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
    similarity_kw: Additional parameters to pass to the similarity calculation function.

The `group_name` has three built-in splitting strategies, all of which use `SentenceSplitter` for splitting, with the difference being in the chunk size:

- CoarseChunk: Chunk size is 1024, with an overlap length of 100
- MediumChunk: Chunk size is 256, with an overlap length of 25
- FineChunk: Chunk size is 128, with an overlap length of 12
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
    similarity_kw: 传递给 similarity 计算函数的其它参数。

其中 `group_name` 有三个内置的切分策略，都是使用 `SentenceSplitter` 做切分，区别在于块大小不同：

- CoarseChunk: 块大小为 1024，重合长度为 100
- MediumChunk: 块大小为 256，重合长度为 25
- FineChunk: 块大小为 128，重合长度为 12
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
''')

# ---------------------------------------------------------------------------- #

# rag/transform.py

add_english_doc('SentenceSplitter', '''
Split sentences into chunks of a specified size. You can specify the size of the overlap between adjacent chunks.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
''')

add_chinese_doc('SentenceSplitter', '''
将句子拆分成指定大小的块。可以指定相邻块之间重合部分的大小。

Args:
    chunk_size (int): 拆分之后的块大小
    chunk_overlap (int): 相邻两个块之间重合的内容长度
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
''')

add_chinese_doc('LLMParser', '''
一个文本摘要和关键词提取器，负责分析用户输入的文本，并根据请求任务提供简洁的摘要或提取相关关键词。

Args:
    llm (TrainableModule): 可训练的模块
    language (str): 语言种类，目前只支持中文（zh）和英文（en）
    task_type (str): 目前支持两种任务：摘要（summary）和关键词抽取（keywords）。
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

add_chinese_doc('rag.DocListManager._delete_files', """\
从数据库中删除指定的文件。

Args:
    file_ids (list of str): 要删除的文件ID列表。
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

add_chinese_doc('rag.DocListManager.update_kb_group_file_status', """\
更新指定知识库分组中文件的状态。

Args:
    file_ids (str or list of str): 文件ID列表。
    status (str): 新的文件状态。
    group (str, optional): 知识库分组名称。默认为None。
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

add_english_doc('rag.DocListManager._delete_files', """\
Deletes specified files from the database.

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

add_english_doc('rag.DocListManager.update_file_status', """\
Updates the status of specified files.

Args:
    file_ids (list of str): List of file IDs to update.
    status (str): The new file status.
""")

add_english_doc('rag.DocListManager.update_kb_group_file_status', """\
Updates the status of files in a specified knowledge base group.

Args:
    file_ids (str or list of str): List of file IDs.
    status (str): The new file status.
    group (str, optional): Name of the knowledge base group. Defaults to None.
""")

add_english_doc('rag.DocListManager.release', """\
Releases the resources of the current manager.
""")

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

add_chinese_doc('ToolManager', '''\
ToolManager是一个工具管理类，用于提供工具信息和工具调用给function call。

此管理类构造时需要传入工具名字符串列表。此处工具名可以是LazyLLM提供的，也可以是用户自定义的，如果是用户自定义的，首先需要注册进LazyLLM中才可以使用。在注册时直接使用 `fc_register` 注册器，该注册器已经建立 `tool` group，所以使用该工具管理类时，所有函数都统一注册进 `tool` 分组即可。待注册的函数需要对函数参数进行注解，并且需要对函数增加功能描述，以及参数类型和作用描述。以方便工具管理类能对函数解析传给LLM使用。

Args:
    tools (List[str]): 工具名称字符串列表。
''')

add_english_doc('ToolManager', '''\
ToolManager is a tool management class used to provide tool information and tool calls to function call.

When constructing this management class, you need to pass in a list of tool name strings. The tool name here can be provided by LazyLLM or user-defined. If it is user-defined, it must first be registered in LazyLLM before it can be used. When registering, directly use the `fc_register` registrar, which has established the `tool` group, so when using the tool management class, all functions can be uniformly registered in the `tool` group. The function to be registered needs to annotate the function parameters, and add a functional description to the function, as well as the parameter type and function description. This is to facilitate the tool management class to parse the function and pass it to LLM for use.

Args:
    tools (List[str]): A list of tool name strings.
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

add_chinese_doc('ReactAgent', '''\
ReactAgent是按照 `Thought->Action->Observation->Thought...->Finish` 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM 使用的工具名称列表。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
''')

add_english_doc('ReactAgent', '''\
ReactAgent follows the process of `Thought->Action->Observation->Thought...->Finish` step by step through LLM and tool calls to display the steps to solve user questions and the final answer to the user.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[str]): A list of tool names for LLM to use.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
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
''')

add_english_doc('PlanAndSolveAgent', '''\
PlanAndSolveAgent consists of two components. First, the planner breaks down the entire task into smaller subtasks, then the solver executes these subtasks according to the plan, which may involve tool calls, and finally returns the answer to the user.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and sovler share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
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
''')

add_english_doc('ReWOOAgent', '''\
ReWOOAgent consists of three parts: Planer, Worker and Solver. The Planner uses predictive reasoning capabilities to create a solution blueprint for a complex task; the Worker interacts with the environment through tool calls and fills in actual evidence or observations into instructions; the Solver processes all plans and evidence to develop a solution to the original task or problem.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and sovler share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
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
    examples (list[list]): extro examples，format is `[[query, intent], [query, intent], ...]`.
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

add_chinese_doc(
    "SqlManager",
    """\
SqlManager是与数据库进行交互的专用工具。它提供了连接数据库，设置、创建、检查数据表，插入数据，执行查询的方法。

Arguments:
    db_type (str): 目前仅支持"PostgreSQL"，后续会增加"MySQL", "MS SQL"
    user (str): username
    password (str): password
    host (str): 主机名或IP
    port (int): 端口号
    db_name (str): 数据仓库名
    tables_info_dict (dict): 数据表的描述
    options_str (str): k1=v1&k2=v2形式表示的选项设置
""",
)

add_english_doc(
    "SqlManager",
    """\
SqlManager is a specialized tool for interacting with databases.
It provides methods for creating tables, executing queries, and performing updates on databases.

Arguments:
    db_type (str): Currently only "PostgreSQL" is supported, with "MySQL" and "MS SQL" to be added later.
    user (str): Username for connection
    password (str): Password for connection
    host (str): Hostname or IP
    port (int): Port number
    db_name (str): Name of the database
    tables_info_dict (dict): Description of the data tables
    options_str (str): Options represented in the format k1=v1&k2=v2
""",
)

add_example(
    "SqlManager",
    """\
    >>> from lazyllm.tools import SqlManager
    >>> import uuid
    >>> # !!!NOTE!!!: COPY class SqlEgsData definition from tests/charge_tests/utils.py then Paste here.
    >>> db_filepath = "personal.db"
    >>> with open(db_filepath, "w") as _:
        pass
    >>> sql_manager = SQLiteManger(filepath, SqlEgsData.TEST_TABLES_INFO)
    >>> # Altert: If using online database, ask administrator about those value: db_type, username, password, host, port, database
    >>> # sql_manager = SqlManager(db_type, username, password, host, port, database, SqlEgsData.TEST_TABLES_INFO)
    >>>
    >>> for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
    ...     sql_manager.execute_sql_update(insert_script)
    >>> str_results = sql_manager.get_query_result_in_json(SqlEgsData.TEST_QUERY_SCRIPTS)
    >>> print(str_results)
""",
)

add_chinese_doc(
    "SqlManager.reset_tables",
    """\
根据描述表结构的字典设置SqlManager所使用的数据表。注意：若表在数据库中不存在将会自动创建，若存在则会校验所有字段的一致性。
字典格式关键字示例如下。

字典中有3个关键字为可选项：表及列的comment默认为空, is_primary_key默认为False但至少应有一列为True, nullable默认为True
{"tables":
    [
        {
            "name": f"employee",
            "comment": "employee information",
            "columns": [
                {
                    "name": "employee_id",
                    "data_type": "Integer",
                    "comment": "empoloyee work number",
                    "nullable": False,
                    "is_primary_key": True,
                },
                {"name": "name", "data_type": "String", "comment": "employee's name", "nullable": False},
                {"name": "department", "data_type": "String", "comment": "employee's department", "nullable": False},
            ],
        },
        {
            ....
        }
    ]
}
""",
)

add_english_doc(
    "SqlManager.reset_tables",
    """\
Set the data tables used by SqlManager according to the dictionary describing the table structure.
Note that if the table does not exist in the database, it will be automatically created, and if it exists, all field consistencies will be checked.
The dictionary format keyword example is as follows.

There are three optional keywords in the dictionary: "comment" for the table and columns defaults to empty, "is_primary_key" defaults to False,
but at least one column should be True, and "nullable" defaults to True.
{"tables":
    [
        {
            "name": f"employee",
            "comment": "employee information",
            "columns": [
                {
                    "name": "employee_id",
                    "data_type": "Integer",
                    "comment": "empoloyee work number",
                    "nullable": False,
                    "is_primary_key": True,
                },
                {"name": "name", "data_type": "String", "comment": "employee's name", "nullable": False},
                {"name": "department", "data_type": "String", "comment": "employee's department", "nullable": False},
            ],
        },
        {
            ....
        }
    ]
}
""",
)

add_chinese_doc(
    "SqlManager.check_connection",
    """\
检查当前SqlManager的连接状态。

**Returns:**\n
- bool: 连接成功(True), 连接失败(False)
- str: 连接成功为"Success" 否则为具体的失败信息.
""",
)

add_english_doc(
    "SqlManager.check_connection",
    """\
Check the current connection status of the SqlManager.

**Returns:**\n
- bool: True if the connection is successful, False if it fails.
- str: "Success" if the connection is successful; otherwise, it provides specific failure information.
""",
)

add_chinese_doc(
    "SqlManager.reset_tables",
    """\
根据提供的表结构设置数据库链接。
若数据库中已存在表项则检查一致性，否则创建数据表

Args:
    tables_info_dict (dict): 数据表的描述

**Returns:**\n
- bool: 设置成功(True), 设置失败(False)
- str: 设置成功为"Success" 否则为具体的失败信息.
""",
)

add_english_doc(
    "SqlManager.reset_tables",
    """\
Set database connection based on the provided table structure.
Check consistency if the table items already exist in the database, otherwise create the data table.

Args:
    tables_info_dict (dict): Description of the data tables

**Returns:**\n
- bool: True if set successfully, False if set failed
- str: "Success" if set successfully, otherwise specific failure information.

""",
)

add_chinese_doc(
    "SqlManager.get_query_result_in_json",
    """\
执行SQL查询并返回JSON格式的结果。
""",
)

add_english_doc(
    "SqlManager.get_query_result_in_json",
    """\
Executes a SQL query and returns the result in JSON format.
""",
)

add_chinese_doc(
    "SqlManager.execute_sql_update",
    """\
在SQLite数据库上执行SQL插入或更新脚本。
""",
)

add_english_doc(
    "SqlManager.execute_sql_update",
    """\
Execute insert or update script.
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
    >>> from lazyllm.tools import SQLiteManger, SqlCall
    >>> sql_tool = SQLiteManger("personal.db")
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
