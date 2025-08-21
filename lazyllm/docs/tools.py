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

# functions for lazyllm.tools.agent
add_agent_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.tools.agent)
add_agent_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.tools.agent)
add_agent_example = functools.partial(utils.add_example, module=lazyllm.tools.agent)

# ---------------------------------------------------------------------------- #

# classifier/intent_classifier.py

add_chinese_doc('IntentClassifier', '''\
意图分类模块，用于根据输入文本在给定的意图列表中进行分类。  
支持中英文自动选择提示模板，并可通过示例、提示、约束和注意事项增强分类效果。

Args:
    llm: 用于意图分类的大语言模型实例。
    intent_list (list): 可选，意图类别列表，例如 ["聊天", "天气", "问答"]。
    prompt (str): 可选，自定义提示语，插入到系统提示模板中。
    constrain (str): 可选，分类约束条件说明。
    attention (str): 可选，提示注意事项。
    examples (list[list[str, str]]): 可选，分类示例列表，每个元素为 [输入文本, 标签]。
    return_trace (bool): 是否返回执行过程的 trace，默认为 False。
''')

add_english_doc('IntentClassifier', '''\
Intent classification module that classifies input text into a given intent list.  
Supports automatic selection of Chinese or English prompt templates, and allows enhancement through examples, prompt text, constraints, and attention notes.

Args:
    llm: The large language model instance used for intent classification.
    intent_list (list): Optional, list of intent categories, e.g., ["chat", "weather", "QA"].
    prompt (str): Optional, custom prompt inserted into the system prompt template.
    constrain (str): Optional, classification constraint description.
    attention (str): Optional, attention notes for classification.
    examples (list[list[str, str]]): Optional, classification examples, each element is [input text, label].
    return_trace (bool): Whether to return execution trace. Default is False.
''')


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


add_chinese_doc('IntentClassifier.intent_promt_hook', '''\
意图分类的预处理 Hook。  
将输入文本与意图列表打包为 JSON，并生成历史对话信息字符串。

Args:
    input (str | List | Dict | None): 输入文本，仅支持字符串类型。
    history (List): 历史对话记录，默认为空列表。
    tools (List[Dict] | None): 工具信息，可选。
    label (str | None): 标签，可选。

**Returns**\n
- tuple: (输入数据字典, 历史记录列表, 工具信息, 标签)
''')

add_english_doc('IntentClassifier.intent_promt_hook', '''\
Pre-processing hook for intent classification.  
Packages the input text and intent list into JSON and generates a string of conversation history.

Args:
    input (str | List | Dict | None): The input text, only string type is supported.
    history (List): Conversation history, default empty list.
    tools (List[Dict] | None): Optional tool information.
    label (str | None): Optional label.

**Returns**\n
- tuple: (input data dict, history list, tools, label)
''')

add_chinese_doc('IntentClassifier.post_process_result', '''\
意图分类结果的后处理。  
如果结果在意图列表中则直接返回，否则返回意图列表的第一个元素。

Args:
    input (str): 分类模型输出结果。

**Returns**\n
- str: 最终的分类标签。
''')

add_english_doc('IntentClassifier.post_process_result', '''\
Post-processing of intent classification result.  
Returns the result directly if it is in the intent list, otherwise returns the first element of the intent list.

Args:
    input (str): Output result from the classification model.

**Returns**\n
- str: The final classification label.
''')

# rag/document.py

add_english_doc('Document', '''\
Initialize a document module with an optional user interface.

This constructor initializes a document module that can have an optional user interface. If the user interface is enabled, it also provides a UI to manage the document operation interface and offers a web page for user interface interaction.

Args:
    dataset_path (str): The path to the dataset directory. This directory should contain the documents to be managed by the document module.
    embed (Optional[Union[Callable, Dict[str, Callable]]]): The object used to generate document embeddings. If you need to generate multiple embeddings for the text, you need to specify multiple embedding models in a dictionary format. The key identifies the name corresponding to the embedding, and the value is the corresponding embedding model.
    create_ui (bool): [Deprecated] Whether to create a user interface. Use 'manager' parameter instead.
    manager (bool, optional): A flag indicating whether to create a user interface for the document module. Defaults to False.
    server (Union[bool, int]): Server configuration. True for default server, False for no server, or an integer port number for custom server.
    name (Optional[str]): Name identifier for this document collection. Required for cloud services.
    launcher (optional): An object or function responsible for launching the server module. If not provided, the default asynchronous launcher from `lazyllm.launchers` is used (`sync=False`).
    doc_fields (optional): Configure the fields that need to be stored and retrieved along with their corresponding types (currently only used by the Milvus backend).
    doc_files (Optional[List[str]]): List of temporary document files (alternative to dataset_path).When used, dataset_path must be None and only map store is supported.
    store_conf (optional): Configure which storage backend, MapStore is the default choice.      
''')

add_chinese_doc('Document', '''\
初始化一个具有可选用户界面的文档模块。

此构造函数初始化一个可以有或没有用户界面的文档模块。如果启用了用户界面，它还会提供一个ui界面来管理文档操作接口，并提供一个用于用户界面交互的网页。

Args:
    dataset_path (str): 数据集目录的路径。此目录应包含要由文档模块管理的文档。
    embed (Optional[Union[Callable, Dict[str, Callable]]]): 用于生成文档 embedding 的对象。如果需要对文本生成多个 embedding，此处需要通过字典的方式指定多个 embedding 模型，key 标识 embedding 对应的名字, value 为对应的 embedding 模型。
    create_ui (bool):[已弃用] 是否创建用户界面。请改用'manager'参数
    manager (bool, optional): 指示是否为文档模块创建用户界面的标志。默认为 False。
    server (Union[bool, int]):服务器配置。True表示默认服务器，False表示已指定端口号作为自定义服务器
    name (Optional[str]):文档集合的名称标识符。云服务模式下必须提供
    launcher (optional): 负责启动服务器模块的对象或函数。如果未提供，则使用 `lazyllm.launchers` 中的默认异步启动器 (`sync=False`)。            
    doc_files (Optional[List[str]]):临时文档文件列表（dataset_path的替代方案）。使用时dataset_path必须为None且仅支持map存储类型
    store_conf (optional): 配置使用哪种存储后端, 默认使用MapStore将切片数据存于内存中。
''')

add_example('Document', '''\
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)  # or documents = Document(dataset_path='your_doc_path', embed={"key": m}, manager=False)
>>> m1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
>>> document1 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, manager=False)

>>> store_conf = {
>>>     "segment_store": {
>>>         "type": "map",
>>>         "kwargs": {
>>>             "uri": "/tmp/tmp_segments.db",
>>>         },
>>>     },
>>>     "vector_store": {
>>>         "type": "milvus",
>>>         "kwargs": {
>>>             "uri": "/tmp/tmp_milvus.db",
>>>             "index_kwargs": {
>>>                 "index_type": "FLAT",
>>>                 "metric_type": "COSINE",
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
    group (str): The name of the node group for which to find the parent.
''')

add_chinese_doc('Document.find_parent', '''
查找指定节点的父节点。

Args:
    group (str): 需要查找的节点组名称
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
    group (str): The name of the node group for which to find the children.
''')

add_chinese_doc('Document.find_children', '''
查找指定节点的子节点。

Args:
    group (str): 需要查找的节点组名称
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

add_chinese_doc('rag.readers.PandasCSVReader', '''\
用于读取 CSV 文件并使用 pandas 进行解析。

Args:
    concat_rows (bool): 是否将所有行拼接为一个文本块，默认为 True。
    col_joiner (str): 列之间的连接符。
    row_joiner (str): 行之间的连接符。
    pandas_config (Optional[Dict]): pandas.read_csv 的可选配置项。
    return_trace (bool): 是否返回处理过程的 trace。
''')

add_english_doc('rag.readers.PandasCSVReader', '''\
Reader for parsing CSV files using pandas.

Args:
    concat_rows (bool): Whether to concatenate all rows into a single text block. Default is True.
    col_joiner (str): String used to join column values.
    row_joiner (str): String used to join rows.
    pandas_config (Optional[Dict]): Optional config for pandas.read_csv.
    return_trace (bool): Whether to return the processing trace.
''')

add_chinese_doc('rag.readers.PandasExcelReader', '''\
用于读取 Excel 文件（.xlsx），并将内容提取为文本。

Args:
    concat_rows (bool): 是否将所有行拼接为一个文本块。
    sheet_name (Optional[str]): 要读取的工作表名称。若为 None，则读取所有工作表。
    pandas_config (Optional[Dict]): pandas.read_excel 的可选配置项。
    return_trace (bool): 是否返回处理过程的 trace。
''')

add_english_doc('rag.readers.PandasExcelReader', '''\
Reader for extracting text content from Excel (.xlsx) files.

Args:
    concat_rows (bool): Whether to concatenate all rows into a single block.
    sheet_name (Optional[str]): Name of the sheet to read. If None, all sheets will be read.
    pandas_config (Optional[Dict]): Optional config for pandas.read_excel.
    return_trace (bool): Whether to return the processing trace.
''')

add_chinese_doc('rag.readers.PDFReader', '''\
用于读取 PDF 文件并提取其中的文本内容。

Args:
    return_full_document (bool): 是否将整份 PDF 合并为一个文档节点。若为 False，则每页作为一个节点。
    return_trace (bool): 是否返回处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.PDFReader', '''\
Reader for extracting text content from PDF files.

Args:
    return_full_document (bool): Whether to merge the entire PDF into a single document node. If False, each page becomes a separate node.
    return_trace (bool): Whether to return the processing trace. Default is True.
''')

add_chinese_doc('rag.readers.PPTXReader', '''\
用于解析 PPTX（PowerPoint）文件的读取器，能够提取幻灯片中的文本，并对嵌入图像进行视觉描述生成。

Args:
    return_trace (bool): 是否记录处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.PPTXReader', '''\
Reader for PPTX (PowerPoint) files. Extracts text from slides and generates captions for embedded images using a vision-language model.

Args:
    return_trace (bool): Whether to record the processing trace. Default is True.
''')

add_chinese_doc('rag.readers.VideoAudioReader', '''\
用于从视频或音频文件中提取语音内容的读取器，依赖 OpenAI 的 Whisper 模型进行语音识别。

Args:
    model_version (str): Whisper 模型的版本（如 "base", "small", "medium", "large"），默认为 "base"。
    return_trace (bool): 是否返回处理过程的 trace，默认为 True。
''')

add_english_doc('rag.readers.VideoAudioReader', '''\
Reader for extracting speech content from video or audio files using OpenAI's Whisper model for transcription.

Args:
    model_version (str): Whisper model version (e.g., "base", "small", "medium", "large"). Default is "base".
    return_trace (bool): Whether to return the processing trace. Default is True.
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

# DocInfoSchemaAnalyser.analyse_info_schema
add_chinese_doc('rag.doc_to_db.DocInfoSchemaAnalyser.analyse_info_schema', '''\
分析文档信息模式的方法，用于从指定类型的文档中提取关键信息字段的结构定义。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于生成信息模式的LLM模型
    doc_type (str): 文档类型，用于指导LLM生成相应的信息模式
    doc_paths (list[str]): 文档路径列表，用于分析的信息来源

**Returns:**\n
- DocInfoSchema: 包含关键信息字段定义的模式列表，每个字段包含key、desc、type三个属性
''')

add_english_doc('rag.doc_to_db.DocInfoSchemaAnalyser.analyse_info_schema', '''\
Method for analyzing document information schema, used to extract structural definitions of key information fields from documents of a specified type.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): LLM model used to generate information schema
    doc_type (str): Document type, used to guide the LLM in generating corresponding information schema
    doc_paths (list[str]): List of document paths, used as information sources for analysis

**Returns:**\n
- DocInfoSchema: List of schema containing key information field definitions, each field includes key, desc, and type attributes
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

add_chinese_doc('rag.doc_to_db.DocInfoExtractor.extract_doc_info', '''\
根据提供的字段结构（schema）从指定文档中抽取具体的关键信息值。

该方法使用大语言模型分析文档内容，根据预定义的字段结构提取相应的信息值，返回格式为 key-value 字典。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于文档信息抽取的大语言模型。
    doc_path (str): 要分析的文档路径。
    info_schema (DocInfoSchema): 字段结构定义，包含需要提取的字段信息。
    extra_desc (str, optional): 额外的描述信息，用于指导信息抽取。默认为空字符串。

Returns:
    dict: 提取出的关键信息字典，键为字段名，值为对应的信息值。
''')

add_english_doc('rag.doc_to_db.DocInfoExtractor.extract_doc_info', '''\
Extracts specific key information values from a document according to a provided schema.

This method uses a large language model to analyze document content and extract corresponding information values based on predefined field structure, returning a key-value dictionary.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): The large language model used for document information extraction.
    doc_path (str): Path to the document to be analyzed.
    info_schema (DocInfoSchema): Field structure definition containing the information to be extracted.
    extra_desc (str, optional): Additional description information to guide the extraction process. Defaults to empty string.

Returns:
    dict: Extracted key information dictionary with field names as keys and corresponding information values as values.
''')

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
获取HTTP响应的内容类型。

从响应头中提取 'content-type' 字段的值，用于判断响应内容的类型。

Returns:
    str: 响应的内容类型，如果未找到则返回空字符串。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
Get the content type of the HTTP response.

Extracts the 'content-type' field value from the response headers to determine the type of response content.

Returns:
    str: The content type of the response, or empty string if not found.
''')

add_example('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
>>> from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
>>> import httpx
>>> response = httpx.Response(200, headers={'content-type': 'application/json'})
>>> http_response = HttpExecutorResponse(response)
>>> content_type = http_response.get_content_type()
>>> print(content_type)
... 'application/json'
''')

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
从HTTP响应中提取文件内容。

如果响应内容类型是文件相关类型（如图片、音频、视频），则提取文件的内容类型和二进制数据。

Returns:
    tuple[str, bytes]: 包含内容类型和文件二进制数据的元组。如果不是文件类型，则返回空字符串和空字节。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
Extract file content from HTTP response.

If the response content type is file-related (such as image, audio, video), extracts the content type and binary data of the file.

Returns:
    tuple[str, bytes]: A tuple containing the content type and binary data of the file. If not a file type, returns empty string and empty bytes.
''')

add_example('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
>>> from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
>>> import httpx
>>> # 模拟图片响应
>>> response = httpx.Response(200, headers={'content-type': 'image/jpeg'}, content=b'fake_image_data')
>>> http_response = HttpExecutorResponse(response)
>>> content_type, file_data = http_response.extract_file()
>>> print(content_type)
... 'image/jpeg'
>>> print(len(file_data))
... 15
>>> # 模拟JSON响应
>>> response = httpx.Response(200, headers={'content-type': 'application/json'}, content=b'{"key": "value"}')
>>> http_response = HttpExecutorResponse(response)
>>> content_type, file_data = http_response.extract_file()
>>> print(content_type)
... ''
>>> print(file_data)
... b''
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

add_chinese_doc('rag.readers.MineruPDFReader', '''\
用于通过 MinerU 服务解析 PDF 文件内容的模块。支持上传文件或通过 URL 方式调用解析接口，解析结果经过回调函数处理成文档节点列表。

Args:
    url (str): MineruPDFReader 服务的接口 URL。
    upload_mode (bool): 是否采用文件上传模式调用接口，默认为 False，即通过 JSON 请求文件路径。
    extract_table (bool): 是否提取表格，默认为 True。
    extract_formula (bool): 是否提取公式，默认为 True。
    split_doc (bool): 是否分割文档，默认为 True。
    post_func (Optional[Callable]): 后处理函数。
''')

add_english_doc('rag.readers.MineruPDFReader', '''\
Module to parse PDF content via the MineruPDFReader service. Supports file upload or URL-based parsing, with a callback to process the parsed elements into document nodes.

Args:
    url (str): The MineruPDFReader service API URL.
    upload_mode (bool): Whether to use file upload mode for the API call. Default is False, meaning JSON request with file path.
    extract_table (bool): Whether to extract tables. Default is True.
    extract_formula (bool): Whether to extract formulas. Default is True.
    split_doc (bool): Whether to split the document. Default is True.
    post_func (Optional[Callable]): Post-processing function.
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
ChromadbStore is a vector-capable implementation of LazyLLMStoreBase, leveraging ChromaDB for persistence and vector search.

Args:
    dir (Optional[str]): Filesystem path for on-disk ChromaDB storage. If provided, a PersistentClient will be used.
    host (Optional[str]): Hostname for ChromaDB HTTP server. Used if `dir` is not set.
    port (Optional[int]): Port number for ChromaDB HTTP server. Used if `dir` is not set.
    index_kwargs (Optional[Union[Dict, List]]): Configuration parameters for ChromaDB collections, e.g., index type and metrics.
    client_kwargs (Optional[Dict]): Additional keyword arguments passed to the ChromaDB client constructor.
''')

add_chinese_doc('rag.store.ChromadbStore', '''
ChromadbStore 是基于 ChromaDB 的向量存储实现，继承自 LazyLLMStoreBase，支持向量写入、检索与持久化。

Args:
    dir (Optional[str]): 本地持久化存储目录，优先使用 PersistentClient 模式。
    host (Optional[str]): HTTP 访问模式下的 ChromaDB 服务主机名。
    port (Optional[int]): HTTP 模式下的 ChromaDB 服务端口。
    index_kwargs (Optional[Union[Dict, List]]): Collection 配置参数，如索引类型、度量方式等。
    client_kwargs (Optional[Dict]): 传递给 ChromaDB 客户端的额外参数。
''')

add_english_doc('rag.store.ChromadbStore.dir', '''
Directory property of the store.

Returns:
    Optional[str]: Normalized directory path ending with a slash, or None if not set.
''')

add_chinese_doc('rag.store.ChromadbStore.dir', '''
存储目录属性。

Returns:
    Optional[str]: 以斜杠结尾的目录路径，若未配置则返回 None。
''')

add_english_doc('rag.store.ChromadbStore.connect', '''
Initialize the ChromaDB client and configure embedding and metadata settings.

Args:
    embed_dims (Dict[str, int]): Dimensions for each embedding key.
    embed_datatypes (Dict[str, DataType]): Data types for global metadata fields.
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Descriptions of global metadata fields.
''')

add_chinese_doc('rag.store.ChromadbStore.connect', '''
初始化 ChromaDB 客户端并配置向量化及元数据相关设定。

Args:
    embed_dims (Dict[str, int]): 每个嵌入键对应的向量维度。
    embed_datatypes (Dict[str, DataType]): 全局元数据字段的数据类型。
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): 全局元数据字段的描述。
''')

add_english_doc('rag.store.ChromadbStore.upsert', '''
Insert or update a batch of records(segment's uid and vectors) into ChromaDB.

Args:
    collection_name (str): Logical name for the collection.
    data (List[dict]): List of documents.

Returns:
    bool: True if operation succeeds, False otherwise.
''')

add_chinese_doc('rag.store.ChromadbStore.upsert', '''
批量写入或更新记录（切片的id及向量数据）到 ChromaDB。

Args:
    collection_name (str): 集合名称。
    data (List[dict]): 文档切片数据列表。

Returns:
    bool: 操作成功返回 True，否则 False。
''')

add_english_doc('rag.store.ChromadbStore.delete', '''
Delete an entire collection or specific records.

Args:
    collection_name (str): Name of the collection.
    criteria (Optional[dict]): If None, drop the collection. Otherwise, filter dict to delete matching records (e.x. delete by doc_id/uid/kb_id).

Returns:
    bool: True if deletion succeeds, False otherwise.
''')

add_chinese_doc('rag.store.ChromadbStore.delete', '''
删除整个集合或指定记录。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 若为 None，删除整个集合；否则按条件删除匹配记录（例如按照切片id、切片所属文件id、切片所属知识库id删除）。

Returns:
    bool: 删除成功返回 True，否则 False。
''')

add_english_doc('rag.store.ChromadbStore.get', '''
Retrieve records matching criteria.

Args:
    collection_name (str): Name of the collection.
    criteria (Optional[dict]): Filter conditions such as primary key or metadata (docid/kb_id).

Returns:
    List[dict]: Each dict contains 'uid' and 'embedding'.
''')

add_chinese_doc('rag.store.ChromadbStore.get', '''
根据条件检索记录。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 过滤条件，如主键或元数据（例如文档id或知识库id）。

Returns:
    List[dict]: 每项包含 'uid' 和 'embedding'。
''')

add_english_doc('rag.store.ChromadbStore.search', '''
Perform a vector similarity search.

Args:
    collection_name (str): Collection to query.
    query_embedding (List[float]): Vector to search with.
    embed_key (str): Which embedding to use.
    topk (int): Number of top results to return.
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): Metadata filter conditions.

Returns:
    List[dict]: Each dict has 'uid' and 'score' (similarity).
''')

add_chinese_doc('rag.store.ChromadbStore.search', '''
执行向量相似度检索。

Args:
    collection_name (str): 要查询的集合名称。
    query_embedding (List[float]): 用于检索的向量。
    embed_key (str): 使用的向量模型的key。
    topk (int): 返回的结果数量。
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): 元数据过滤条件。

Returns:
    List[dict]: 每项包含 'uid' 及 'score'（相似度）。
''')

add_english_doc('rag.store.MilvusStore', '''
Vector store implementation based on Milvus, inheriting from StoreBase. Supports vector insertion, deletion, flexible querying (including scalar filtering).

Args:
    uri (str): Milvus connection URI (e.g., "tcp://localhost:19530"). If scheme is local file path, uses milvus-lite version; otherwise remote (need to set up a milvus service, e.x. standalone/distributed version).
    db_name (str): Database name to use in Milvus. Defaults to "lazyllm".
    index_kwargs (Optional[Union[Dict, List]]): Index creation parameters (e.g., {"index_type": "IVF_FLAT", "metric_type": "COSINE"} or a list of per-embed-key configs).
    client_kwargs (Optional[Dict]): Additional keyword arguments for milvus client.
''')

add_chinese_doc('rag.store.MilvusStore', '''
基于 Milvus 的向量存储实现，继承自 StoreBase。支持向量写入、删除、相似度检索，兼容标量过滤。

Args:
    uri (str): Milvus 连接 URI（如 "tcp://localhost:19530"）。如果为本地路径则使用milvus-lite，否则为远程模式（需要独立部署milvus服务，例如standalone/distributed版本）。
    db_name (str): Milvus 中使用的数据库名称，默认为 "lazyllm"。
    index_kwargs (Optional[Union[Dict, List]]): 索引创建参数（例如 {"index_type": "IVF_FLAT", "metric_type": "CONSINE"} ，支持按向量模型的key配置列表）。
    client_kwargs (Optional[Dict]): 传递给 milvus 客户端的额外参数。
''')

add_english_doc('rag.store.MilvusStore.dir', '''
Local storage directory derived from URI if running embedded. Returns None when using remote Milvus.

Returns:
    Optional[str]: Directory path for local milvus.db file, or None if remote.
''')

add_chinese_doc('rag.store.MilvusStore.dir', '''
存储目录属性，基于 URI 推断。远程模式返回 None。

Returns:
    Optional[str]: 本地 milvus.db 文件的目录路径，或 None。
''')

add_english_doc('rag.store.MilvusStore.connect', '''
Initialize Milvus client, pass in embedding model parameters and global metadata descriptions.

Args:
    embed_dims (Dict[str, int]): Embedding dimensions per embed key.
    embed_datatypes (Dict[str, DataType]): Data types for each embed key.
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Descriptions for metadata fields.
''')

add_chinese_doc('rag.store.MilvusStore.connect', '''
初始化 Milvus 客户端，传入向量化模型参数和全局元数据描述。

Args:
    embed_dims (Dict[str, int]): 每个嵌入键对应的向量维度。
    embed_datatypes (Dict[str, DataType]): 每个嵌入键的数据类型。
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): 全局元数据字段的描述。
''')

add_english_doc('rag.store.MilvusStore.upsert', '''
Insert or update a batch of segment data into the Milvus collection.

Args:
    collection_name (str): Collection name (per embed key grouping).
    data (List[dict]): List of segment data.
Returns:
    bool: True if successful, False otherwise.
''')

add_chinese_doc('rag.store.MilvusStore.upsert', '''
批量写入或更新切片数据到 Milvus 集合。

Args:
    collection_name (str): 集合名称，通常为 "group_embedKey" 格式。
    data (List[dict]): 切片数据列表。
Returns:
    bool: 操作成功返回 True，否则 False。
''')

add_english_doc('rag.store.MilvusStore.delete', '''
Delete entire collection or subset of records by criteria.

Args:
    collection_name (str): Target collection.
    criteria (Optional[dict]): If None, drop the entire collection; otherwise a dict of filters (uid list or metadata conditions).
Returns:
    bool: True if deletion succeeds, False otherwise.
''')

add_chinese_doc('rag.store.MilvusStore.delete', '''
删除整个集合或按条件删除指定记录。

Args:
    collection_name (str): 目标集合名称。
    criteria (Optional[dict]): 若为 None 则删除整个集合；否则按 uid 列表或元数据条件过滤。
Returns:
    bool: 删除成功返回 True，否则 False。
''')

add_english_doc('rag.store.MilvusStore.get', '''
Retrieve records matching primary-key or metadata filters.

Args:
    collection_name (str): Collection to query.
    criteria (Optional[dict]): Dict containing 'uid' list or metadata field filters.
Returns: 
    List[dict]: Each entry contains 'uid' and 'embedding'.
''')

add_chinese_doc('rag.store.MilvusStore.get', '''
检索匹配主键或元数据过滤条件的记录。

Args:
    collection_name (str): 待查询集合。
    criteria (Optional[dict]): 包含 'uid' 列表或元数据字段过滤条件。
Returns:
    List[dict]: 每项包含 'uid' 及 'embedding' 映射。
''')

add_english_doc('rag.store.MilvusStore.search', '''
Perform vector similarity search with optional metadata filtering.

Args:
    collection_name (str): Collection to search.
    query_embedding (List[float]): Query vector.
    topk (int): Number of nearest neighbors.
    filters (Optional[Dict[str, Union[List, Set]]]): Metadata filter map.
    embed_key (str): Which embedding field to use.
Returns:
    List[dict]: Each dict has 'uid' and similarity 'score'.
''')

add_chinese_doc('rag.store.MilvusStore.search', '''
执行向量相似度检索，并可按元数据过滤。

Args:
    collection_name (str): 待搜索集合。
    query_embedding (List[float]): 查询向量。
    topk (int): 返回邻近数量。
    filters (Optional[Dict[str, Union[List, Set]]]): 元数据过滤映射。
    embed_key (str): 使用的嵌入字段。
Returns:
    List[dict]: 每项包含 'uid' 及相似度 'score'。
''')

add_chinese_doc('rag.default_index.DefaultIndex', r'''\ 
默认的索引实现，负责通过 embedding 和文本相似度在底层存储中查询、更新和删除文档节点。支持多种相似度度量方式，并在必要时对查询和节点进行 embedding 计算与更新。

Args:
    embed (Dict[str, Callable]): 用于生成查询和节点 embedding 的字典，key 是 embedding 名称，value 是接收字符串返回向量的函数。
    store (StoreBase): 底层存储，用于持久化和检索 DocNode 节点。
    **kwargs: 预留扩展参数。
''')

add_english_doc('rag.default_index.DefaultIndex', '''\
Default index implementation responsible for querying, updating, and removing document nodes in the underlying store using embedding or text similarity. Supports multiple similarity metrics and performs embedding computation and node updates when needed.

Args:
    embed (Dict[str, Callable]): Mapping of embedding names to functions that generate vector representations from strings.
    store (StoreBase): Underlying storage to persist and retrieve DocNode objects.
    **kwargs: Reserved for future extension.
''')

add_chinese_doc('rag.default_index.DefaultIndex.update', r'''\ 
根据提供的节点列表更新索引中的内容。具体行为由子类或外部实现填充（此处为空实现，需在实际使用中覆盖/扩展）。

Args:
    nodes (List[DocNode]): 需要更新（新增或替换）的文档节点列表。
''')

add_english_doc('rag.default_index.DefaultIndex.update', '''\
Update the index with the given list of document nodes. This is a placeholder implementation and should be provided/extended in concrete usage.

Args:
    nodes (List[DocNode]): Document nodes to add or update in the index.
''')

add_chinese_doc('rag.default_index.DefaultIndex.remove', r'''\ 
从索引中删除指定 UID 的节点，可选指定分组名称以限定作用域。当前为空实现，使用时需要补全逻辑。

Args:
    uids (List[str]): 要删除的节点唯一标识列表。
    group_name (Optional[str]): 可选的分组名称，用于限定删除范围。
''')

add_english_doc('rag.default_index.DefaultIndex.remove', '''\
Remove nodes with specified UIDs from the index. Optionally scoped to a group. This is a no-op placeholder and should be implemented in concrete usage.

Args:
    uids (List[str]): List of unique IDs of nodes to remove.
    group_name (Optional[str]): Optional group name to scope the removal.
''')

add_chinese_doc('rag.default_index.DefaultIndex.query', r'''\ 
执行一次查询，支持 embedding 和文本两种模式，依据相似度函数过滤并返回符合条件的 DocNode 结果。

Args:
    query (str): 原始查询文本。
    group_name (str): 要检索的节点组名称。
    similarity_name (str): 使用的相似度度量名称，必须在 registered_similarities 中注册。
    similarity_cut_off (Union[float, Dict[str, float]]): 相似度阈值或每个 embedding 对应的阈值字典，用于过滤结果。
    topk (int): 每个相似度渠道最多保留的候选数量。
    embed_keys (Optional[List[str]]): 指定用于 embedding 的 key 列表，若为空则使用所有可用 embedding。
    filters (Optional[Dict[str, List]]): 额外的节点过滤器，应用在计算相似度前。
    **kwargs: 传递给相似度函数的额外参数。

**Returns**\n
    - list: List[DocNode]: 经过相似度计算与阈值过滤后去重的文档节点列表。
''')

add_english_doc('rag.default_index.DefaultIndex.query', '''\
Perform a query against the index, supporting both embedding-based and text-based similarity modes. Filters and ranks nodes according to similarity functions and cutoffs.

Args:
    query (str): The raw query string.
    group_name (str): The group name from which to retrieve nodes.
    similarity_name (str): Name of the similarity metric to use; must be registered in registered_similarities.
    similarity_cut_off (Union[float, Dict[str, float]]): Similarity threshold(s) used to filter results; can be a single float or a mapping per embedding.
    topk (int): Maximum number of candidates to keep per similarity channel before final filtering.
    embed_keys (Optional[List[str]]): Specific embedding keys to use; defaults to all available if not provided.
    filters (Optional[Dict[str, List]]): Additional pre-filters applied to nodes before similarity computation.
    **kwargs: Extra keyword arguments forwarded to the similarity function.

**Returns**\n
    - list: List[DocNode]: Deduplicated list of document nodes passing similarity and cutoff criteria.
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
**Detailed explanation of reranker types**

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
    kwargs: 传递给重新排序器实例化的其他关键字参数。

详细解释排序器类型

  - Reranker: 实例化一个具有待排序的文档节点node列表和 query的 SentenceTransformerRerank 重排序器。
  - KeywordFilter: 实例化一个具有指定必需和排除关键字的 KeywordNodePostprocessor。它根据这些关键字的存在或缺失来过滤节点。
''')

add_example('Reranker', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, Reranker, Retriever, DocNode
>>> m = lazyllm.OnlineEmbeddingModule()
>>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
>>> retriever = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
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

**Returns:**\n
- self: For method chaining.
''')

add_chinese_doc('rag.retriever.TempDocRetriever.add_subretriever', '''
添加带搜索配置的子检索器。

Args:
    group (str): 目标节点组名称。
    **kwargs: 检索器参数（如similarity='cosine'）。

**Returns:**\n
- self: 支持链式调用。
''')

add_chinese_doc('rag.document.UrlDocument', '''\
UrlDocument类继承自ModuleBase，用于通过指定的URL和名称管理远程文档资源。  
内部通过lazyllm的UrlModule代理实际调用，支持文档查找、检索和活跃节点分组查询。  

Args:
    url (str): 远程文档资源的访问URL。
    name (str): 当前文档分组名称，用于标识文档分组。
''')

add_english_doc('rag.document.UrlDocument', '''\
UrlDocument class inherits from ModuleBase, used to manage remote document resources by specifying a URL and a name.  
Internally delegates calls to lazyllm's UrlModule, supporting document find, retrieve, and querying active node groups.

Args:
    url (str): Access URL for the remote document resource.
    name (str): Current document group name used to identify the document group.
''')

add_chinese_doc('rag.document.UrlDocument.find', '''\
生成一个部分应用函数，用于在当前文档组中查找指定目标。

Args:
    target (str): 需要查找的目标标识。

**Returns:**\n
- Callable: 调用时会执行查找操作的部分应用函数。
''')

add_english_doc('rag.document.UrlDocument.find', '''\
Creates a partially applied function to find a specified target within the current document group.

Args:
    target (str): The target identifier to find.

**Returns:**\n
- Callable: A partially applied function that executes the find operation when called.
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

add_chinese_doc('rag.doc_processor.DocumentProcessor', """
文档处理器类，用于管理文档的添加、删除和更新操作。

Args:
    server (bool): 是否以服务器模式运行。默认为True。
    port (Optional[int]): 服务器端口号。默认为None。
    url (Optional[str]): 远程服务URL。默认为None。

**说明:**
- 支持异步处理文档任务
- 提供文档元数据更新功能
- 支持任务状态回调通知
- 可配置数据库存储
""")

add_english_doc('rag.doc_processor.DocumentProcessor', """
Document processor class for managing document addition, deletion and update operations.

Args:
    server (bool): Whether to run in server mode. Defaults to True.
    port (Optional[int]): Server port number. Defaults to None.
    url (Optional[str]): Remote service URL. Defaults to None.

**Notes:**
- Supports asynchronous document task processing
- Provides document metadata update functionality
- Supports task status callback notifications
- Configurable database storage
""")

add_example('rag.doc_processor.DocumentProcessor', """
```python
# Create local document processor
processor = DocumentProcessor(server=False)

# Create server mode document processor
processor = DocumentProcessor(server=True, port=8080)

# Create remote document processor
processor = DocumentProcessor(url="http://remote-server:8080")
```
""")

add_chinese_doc('rag.doc_processor.DocumentProcessor.register_algorithm', """
注册算法到文档处理器。

Args:
    name (str): 算法名称，作为唯一标识符。
    store (StoreBase): 存储实例，用于管理文档数据。
    reader (ReaderBase): 读取器实例，用于解析文档内容。
    node_groups (Dict[str, Dict]): 节点组配置信息。
    force_refresh (bool): 是否强制刷新已存在的算法。默认为False。

**说明:**
- 如果算法名称已存在且force_refresh为False，将跳过注册
- 注册成功后可以使用该算法处理文档
""")

add_english_doc('rag.doc_processor.DocumentProcessor.register_algorithm', """
Register an algorithm to the document processor.

Args:
    name (str): Algorithm name as unique identifier.
    store (StoreBase): Storage instance for managing document data.
    reader (ReaderBase): Reader instance for parsing document content.
    node_groups (Dict[str, Dict]): Node group configuration information.
    force_refresh (bool): Whether to force refresh existing algorithm. Defaults to False.

**Notes:**
- If algorithm name exists and force_refresh is False, registration will be skipped
- After successful registration, the algorithm can be used to process documents
""")

add_example('rag.doc_processor.DocumentProcessor.register_algorithm', """
```python
from lazyllm.rag import DocumentProcessor, FileStore, PDFReader

# Create storage and reader instances
store = FileStore(path="./data")
reader = PDFReader()

# Define node group configuration
node_groups = {
    "text": {"transform": "text", "parent": "root"},
    "summary": {"transform": "summary", "parent": "text"}
}

# Register algorithm
processor = DocumentProcessor()
processor.register_algorithm(
    name="pdf_processor",
    store=store,
    reader=reader,
    node_groups=node_groups
)
```
""")

add_chinese_doc('rag.doc_processor.DocumentProcessor.drop_algorithm', """
从文档处理器中移除指定算法。

Args:
    name (str): 要移除的算法名称。
    clean_db (bool): 是否清理相关数据库数据。默认为False。

**说明:**
- 如果算法名称不存在，将输出警告信息
- 移除后该算法将无法继续使用
""")

add_english_doc('rag.doc_processor.DocumentProcessor.drop_algorithm', """
Remove specified algorithm from document processor.

Args:
    name (str): Name of the algorithm to remove.
    clean_db (bool): Whether to clean related database data. Defaults to False.

**Notes:**
- If algorithm name does not exist, a warning message will be output
- After removal, the algorithm will no longer be available
""")

add_example('rag.doc_processor.DocumentProcessor.drop_algorithm', """
```python
# Remove algorithm
processor.drop_algorithm("pdf_processor")

# Remove algorithm and clean database
processor.drop_algorithm("pdf_processor", clean_db=True)
```
""")

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

add_chinese_doc('rag.dataReader.SimpleDirectoryReader.load_file', '''\
load_file(input_file, metadata_genf, file_extractor, encoding='utf-8', pathm=Path, fs=None, metadata=None)

使用指定的 Reader 将单个文件加载为 `DocNode` 列表。

该方法会根据文件名匹配合适的读取器（reader），并遵循以下优先级生成元数据：
`用户提供 > reader 自动生成 > metadata_genf 生成`。支持自定义文件读取器，同时在配置允许的情况下支持回退到原始文本读取。

参数说明：
- input_file (Path): 要读取的文件路径。
- metadata_genf (Callable): 用于根据路径生成元数据的函数。
- file_extractor (Dict[str, Callable]): 文件扩展名与 reader 的映射表。
- encoding (str): 文件读取时使用的文本编码，默认为 "utf-8"。
- pathm (PurePath): 路径处理模块，适用于本地或远程路径。
- fs (AbstractFileSystem): 可选的文件系统对象，支持 fsspec 抽象。
- metadata (Dict): 可选的用户自定义元数据，优先于自动生成。

返回：
- List[DocNode]: 从文件中提取的文档对象列表。
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader.load_file', '''\
load_file(input_file, metadata_genf, file_extractor, encoding='utf-8', pathm=Path, fs=None, metadata=None)

Load a single file into a list of `DocNode` objects using the appropriate reader.

This method supports automatic reader selection based on file extension patterns, and applies a priority order to metadata:
`user > reader > metadata_genf`. It supports both default and user-supplied readers and can fall back to raw text decoding
if enabled in config.

Parameters:
- input_file (Path): Path to the input file.
- metadata_genf (Callable): Function to generate metadata from file path.
- file_extractor (Dict[str, Callable]): Mapping of file extension patterns to reader callables.
- encoding (str): Text encoding to use when reading files. Default is "utf-8".
- pathm (PurePath): Path handling module to support local or remote paths.
- fs (AbstractFileSystem): Optional filesystem abstraction from fsspec.
- metadata (Dict): Optional user-defined metadata to override reader-generated data.

Returns:
- List[DocNode]: List of parsed documents extracted from the file.
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

add_chinese_doc('rag.readers.readerBase.LazyLLMReaderBase', '''
基础文档读取器类，提供了文档加载的基本接口。继承自ModuleBase，使用LazyLLMRegisterMetaClass作为元类。

Args:
    return_trace (bool): 是否返回处理过程的追踪信息。默认为True。

**说明:**
- 提供了惰性加载和普通加载两种方式
- 子类需要实现_lazy_load_data方法
- 支持批量处理文档
- 自动转换为标准化的DocNode格式
''')

add_english_doc('rag.readers.readerBase.LazyLLMReaderBase', '''
Base document reader class that provides basic interfaces for document loading. Inherits from ModuleBase and uses LazyLLMRegisterMetaClass as metaclass.

Args:
    return_trace (bool): Whether to return processing trace information. Defaults to True.

**Notes:**
- Provides both lazy loading and regular loading methods
- Subclasses need to implement _lazy_load_data method
- Supports batch document processing
- Automatically converts to standardized DocNode format
''')

add_example('rag.readers.readerBase.LazyLLMReaderBase', '''
```python
from lazyllm.tools.rag.readers.readerBase import LazyLLMReaderBase
from lazyllm.tools.rag.doc_node import DocNode
from typing import Iterable

class CustomReader(LazyLLMReaderBase):
    def _lazy_load_data(self, file_paths: list, **kwargs) -> Iterable[DocNode]:
        for file_path in file_paths:
            # Process each file and yield DocNode
            content = self._read_file(file_path)
            yield DocNode(
                text=content,
                metadata={"source": file_path}
            )

# Create reader instance
reader = CustomReader(return_trace=True)

# Load documents
documents = reader.forward(file_paths=["doc1.txt", "doc2.txt"])
```
''')


add_chinese_doc('rag.doc_node.QADocNode', '''\
问答文档节点类，用于存储问答对数据。

参数:
    query (str): 问题文本。
    answer (str): 答案文本。
    uid (str): 唯一标识符。
    group (str): 文档组名。
    embedding (Dict[str, List[float]]): 嵌入向量字典。
    parent (DocNode): 父节点引用。
    metadata (Dict[str, Any]): 节点级元数据。
    global_metadata (Dict[str, Any]): 文档级元数据。
    text (str): 节点内容，与query互斥。
''')

add_english_doc('rag.doc_node.QADocNode', '''\
Question-Answer document node class for storing QA pair data.

Args:
    query (str): The question text.
    answer (str): The answer text.
    uid (str): Unique identifier.
    group (str): Document group name.
    embedding (Dict[str, List[float]]): Dictionary of embedding vectors.
    parent (DocNode): Reference to the parent node.
    metadata (Dict[str, Any]): Node-level metadata.
    global_metadata (Dict[str, Any]): Document-level metadata.
    text (str): Node content, mutually exclusive with query.
''')

add_chinese_doc('rag.doc_node.QADocNode.get_text', '''\
获取节点的文本内容。

参数:
    metadata_mode (MetadataMode): 元数据模式，默认为MetadataMode.NONE。
        当设置为MetadataMode.LLM时，返回格式化的问答对。
        其他模式下返回基类的文本格式。

返回值:
    str: 格式化后的文本内容。
''')

add_english_doc('rag.doc_node.QADocNode.get_text', '''\
Get the text content of the node.

Args:
    metadata_mode (MetadataMode): Metadata mode, defaults to MetadataMode.NONE.
        When set to MetadataMode.LLM, returns formatted QA pair.
        For other modes, returns base class text format.

Returns:
    str: The formatted text content.
''')

# ---------------------------------------------------------------------------- #

# rag/transform.py

add_english_doc('SentenceSplitter', '''
Split sentences into chunks of a specified size. You can specify the size of the overlap between adjacent chunks.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
''')

add_chinese_doc('SentenceSplitter', '''
将句子拆分成指定大小的块。可以指定相邻块之间重合部分的大小。

Args:
    chunk_size (int): 拆分之后的块大小
    chunk_overlap (int): 相邻两个块之间重合的内容长度
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
''')

add_chinese_doc('LLMParser', '''
一个文本摘要和关键词提取器，负责分析用户输入的文本，并根据请求任务提供简洁的摘要或提取相关关键词。

Args:
    llm (TrainableModule): 可训练的模块
    language (str): 语言种类，目前只支持中文（zh）和英文（en）
    task_type (str): 目前支持两种任务：摘要（summary）和关键词抽取（keywords）。
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

add_chinese_doc('rag.DocManager.delete_files_from_group', """
删除指定分组中的文件的接口。

Args:
    request (FileGroupRequest): 包含文件ID列表和分组名称的请求参数。

**Returns:**\n
- BaseResponse: 删除操作结果。
""")

add_chinese_doc('rag.DocManager.add_metadata', """
为指定文档添加或更新元数据的接口。

Args:
    add_metadata_request (AddMetadataRequest): 包含文档ID列表和键值对元数据的请求。

**Returns:**\n
- BaseResponse: 操作结果信息。
""")

add_chinese_doc('rag.DocManager.delete_metadata_item', """
删除指定文档的元数据字段或字段值的接口。

Args:
    del_metadata_request (DeleteMetadataRequest): 包含文档ID列表、字段名和键值对删除条件的请求。

**Returns:**\n
- BaseResponse: 操作结果信息。
""")

add_chinese_doc('rag.DocManager.update_or_create_metadata_keys', """
更新或创建文档元数据字段的接口。
Args:
    update_metadata_request (UpdateMetadataRequest): 包含文档ID列表和需更新或新增的键值对元数据。

**Returns:**\n
- BaseResponse: 操作结果信息。
""")

add_chinese_doc('rag.DocManager.reset_metadata', """
重置指定文档的所有元数据字段。

Args:
    reset_metadata_request (ResetMetadataRequest): 包含文档ID列表和新的元数据字典。

**Returns:**\n
- BaseResponse: 操作结果信息。
""")

add_chinese_doc('rag.DocManager.query_metadata', """
查询指定文档的元数据。

Args:
    query_metadata_request (QueryMetadataRequest): 请求参数，包含文档ID和可选的字段名。

**Returns:**\n
- BaseResponse: 若指定了 key 且存在，返回对应字段值；否则返回整个 metadata；key 不存在时报错。
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

add_english_doc('rag.DocManager.delete_files_from_group', """
An endpoint to delete specified files in a group.

Args:
    request (FileGroupRequest): Request containing a list of file IDs and the group name.

**Returns:**\n
- BaseResponse: Deletion operation result.
""")

add_english_doc('rag.DocManager.add_metadata', """
An endpoint to add or update metadata for specified documents.

Args:
    add_metadata_request (AddMetadataRequest): Request containing list of document IDs and key-value metadata.

**Returns:**\n
- BaseResponse: Operation result information.
""")

add_english_doc('rag.DocManager.delete_metadata_item', """
An endpoint to delete metadata fields or field values from specified documents.

Args:
    del_metadata_request (DeleteMetadataRequest): Request containing list of document IDs, field names, and/or deletion rules.

**Returns:**\n
- BaseResponse: Deletion operation result.
""")

add_english_doc('rag.DocManager.update_or_create_metadata_keys', """
An endpoint to update or create metadata fields for specified documents.

Args:
    update_metadata_request (UpdateMetadataRequest): Request containing a list of document IDs and metadata key-value pairs to update or create.

**Returns:**\n
- BaseResponse: Deletion operation result.
""")

add_english_doc('rag.DocManager.reset_metadata', """
An endpoint to reset all metadata fields of specified documents.

Args:
    reset_metadata_request (ResetMetadataRequest): Request containing a list of document IDs and the new metadata dictionary to apply.

**Returns:**\n
- BaseResponse: Deletion operation result.
""")

add_english_doc('rag.DocManager.query_metadata', """
An endpoint to query metadata of a specific document.

Args:
    query_metadata_request (QueryMetadataRequest): Request containing the document ID and an optional metadata field name.

**Returns:**\n
- BaseResponse: Returns the field value if key is specified and exists; otherwise returns full metadata. If the key does not exist, returns an error.
""")
# ---------------------------------------------------------------------------- #

# rag/data_loaders.py

add_english_doc('rag.data_loaders.DirectoryReader', '''\
A directory reader class for loading and processing documents from file directories.

This class provides functionality to read documents from specified directories and convert them into document nodes. It supports both local and global file readers, and can handle different types of documents including images.

Args:
    input_files (Optional[List[str]]): A list of file paths to read. If None, files will be loaded when calling load_data method.
    local_readers (Optional[Dict]): A dictionary of local file readers specific to this instance. Keys are file patterns, values are reader functions.
    global_readers (Optional[Dict]): A dictionary of global file readers shared across all instances. Keys are file patterns, values are reader functions.
''')

add_chinese_doc('rag.data_loaders.DirectoryReader', '''\
用于从文件目录加载和处理文档的目录读取器类。

此类提供从指定目录读取文档并将其转换为文档节点的功能。它支持本地和全局文件读取器，并且可以处理不同类型的文档，包括图像。

Args:
    input_files (Optional[List[str]]): 要读取的文件路径列表。如果为None，文件将在调用load_data方法时加载。
    local_readers (Optional[Dict]): 特定于此实例的本地文件读取器字典。键是文件模式，值是读取器函数。
    global_readers (Optional[Dict]): 在所有实例间共享的全局文件读取器字典。键是文件模式，值是读取器函数。
''')

add_example('rag.data_loaders.DirectoryReader', '''\
>>> from lazyllm.tools.rag.data_loaders import DirectoryReader
>>> from lazyllm.tools.rag.readers import DocxReader, PDFReader
>>> local_readers = {
...     "**/*.docx": DocxReader,
...     "**/*.pdf": PDFReader
>>> }
>>> reader = DirectoryReader(
...     input_files=["path/to/documents"],
...     local_readers=local_readers,
...     global_readers={}
>>> )
>>> documents = reader.load_data()
>>> print(f"加载了 {len(documents)} 个文档")
''')

add_english_doc('rag.data_loaders.DirectoryReader.load_data', '''\
Load and process documents from the specified input files.

This method reads documents from the input files using the configured file readers (both local and global), processes them into document nodes, and optionally separates image nodes from text nodes.

Args:
    input_files (Optional[List[str]]): A list of file paths to read. If None, uses the files specified during initialization.
    metadatas (Optional[Dict]): Additional metadata to associate with the loaded documents.
    split_image_nodes (bool): Whether to separate image nodes from text nodes. If True, returns a tuple of (text_nodes, image_nodes). If False, returns all nodes together.

**Returns:**
- Union[List[DocNode], Tuple[List[DocNode], List[ImageDocNode]]]: If split_image_nodes is False, returns a list of all document nodes. If True, returns a tuple containing text nodes and image nodes separately.

''')

add_chinese_doc('rag.data_loaders.DirectoryReader.load_data', '''\
从指定的输入文件加载和处理文档。

此方法使用配置的文件读取器（本地和全局）从输入文件读取文档，将它们处理成文档节点，并可选地将图像节点与文本节点分离。

Args:
    input_files (Optional[List[str]]): 要读取的文件路径列表。如果为None，使用初始化时指定的文件。
    metadatas (Optional[Dict]): 与加载文档关联的额外元数据。
    split_image_nodes (bool): 是否将图像节点与文本节点分离。如果为True，返回(text_nodes, image_nodes)的元组。如果为False，一起返回所有节点。

**Returns:**
- Union[List[DocNode], Tuple[List[DocNode], List[ImageDocNode]]]: 如果split_image_nodes为False，返回所有文档节点的列表。如果为True，返回包含文本节点和图像节点的元组。
''')

# ---------------------------------------------------------------------------- #

# rag/utils.py
add_chinese_doc('rag.utils.DocListManager', """\
抽象基类，用于管理文档列表和监控文档目录变化。

Args:
    path:要监控的文档目录路径。
    name:管理器名称。
    enable_path_monitoring:启用路径监控。

""")

add_chinese_doc('rag.utils.DocListManager.init_tables', """\
确保数据库表默认分组存在。
""")

add_chinese_doc('rag.utils.DocListManager.delete_files', """\
将与文件关联的知识库条目设为删除中，并由各知识库进行异步删除解析结果及关联记录。

Args:
    file_ids (list of str): 要删除的文件ID列表
""")

add_chinese_doc('rag.utils.DocListManager.table_inited', """\
检查数据库中的 `documents` 表是否已初始化。此方法在访问数据库时确保线程安全。
判断数据库中是否存在 `documents` 表。
返回值:
    bool: 如果 `documents` 表存在，返回 `True`；否则返回 `False`。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保对数据库的安全访问。
    - 通过 `self._db_path` 连接 SQLite 数据库，并使用 `check_same_thread` 配置选项。
    - 执行 SQL 查询：`SELECT name FROM sqlite_master WHERE type='table' AND name='documents'` 来检查表是否存在。
""")

add_chinese_doc('rag.utils.DocListManager.validate_paths', '''\
验证一组文件路径，以确保它们可以被正常处理。
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

add_chinese_doc('rag.utils.DocListManager.update_need_reparsing', '''\
更新 `KBGroupDocuments` 表中某个文档的 `need_reparse` 状态。
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

add_chinese_doc('rag.utils.DocListManager.list_files', """\
从 `documents` 表中列出文件，并支持过滤、限制返回结果以及返回详细信息。
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
""")

add_chinese_doc('rag.utils.DocListManager.get_docs', '''\
从数据库中检索类型为 `KBDocument` 的文档对象，基于提供的文档 ID 列表。

Args:
    doc_ids (List[str]): 要获取的文档 ID 列表。
**Returns:**
    List[KBDocument]: 与提供的文档 ID 对应的 `KBDocument` 对象列表。如果没有找到文档，将返回空列表。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问的安全性。
    - 查询使用 SQL 的 `IN` 子句，通过 `doc_id` 字段进行过滤。
    - 如果 `doc_ids` 为空，函数将直接返回空列表，而不会查询数据库。
''')

add_chinese_doc('rag.utils.DocListManager.set_docs_new_meta', """\
批量更新文档的元数据。

Args:
    doc_meta (Dict[str, dict]): 文档ID到新元数据的映射字典。

""")

add_chinese_doc('rag.utils.DocListManager.fetch_docs_changed_meta', '''\
获取指定组中元数据已更改的文档，并将其 `new_meta` 字段重置为 `None`。
此方法检索元数据已更改（即 `new_meta` 不为 `None`）的所有文档，基于提供的组名。检索后，会将这些文档的 `new_meta` 字段重置为 `None`。

Args:
    group (str): 用于过滤文档的组名。
**Returns:**
    List[DocMetaChangedRow]: 包含文档 `doc_id` 和 `new_meta` 字段的行列表，表示元数据已更改的文档。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保数据库访问安全。
    - 方法通过 SQL `JOIN` 操作连接 `KBDocument` 和 `KBGroupDocuments` 表以检索相关行。
    - 在获取数据后，将受影响行的 `new_meta` 字段更新为 `None`，并将更改提交到数据库。
''')

add_chinese_doc('rag.utils.DocListManager.add_kb_group', """\
添加一个新的知识库分组。

Args:
    name (str): 要添加的分组名称。
""")

add_chinese_doc('rag.utils.DocListManager.list_kb_group_files', '''\
列出指定知识库组中的文件。

Args:
    group (str): 用于过滤文件的 KB 组名。默认为 `None`。
    limit (Optional[int]): 返回的最大文件数量。如果为 `None`，则返回所有匹配的文件。
    details (bool): 返回详细的文件信息或仅返回文件 ID 和路径。
    status (Union[str, List[str]]): 包含在结果中的 KB 组状态或状态列表。默认为所有状态。
    exclude_status (Optional[Union[str, List[str]]): 从结果中排除的 KB 组状态或状态列表。默认为 `None`。
    upload_status (Union[str, List[str]]): 包含在结果中的文档上传状态或状态列表。默认为所有状态。
    exclude_upload_status (Optional[Union[str, List[str]]): 从结果中排除的文档上传状态或状态列表。默认为 `None`。
    need_reparse (Optional[bool]): 过滤需要重新解析的文件或不需要重新解析的文件。默认为 `None`。
**Returns:**:
    List: 如果 `details=False`，返回包含 `(doc_id, path)` 的元组列表。
          如果 `details=True`，返回包含附加元数据的详细行列表。
说明:
    - 方法根据提供的过滤条件动态构建 SQL 查询。
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - 如果 `status` 或 `upload_status` 参数为列表，则会使用 SQL 的 `IN` 子句进行处理。
''')

add_chinese_doc('rag.utils.DocListManager.list_all_kb_group', """\
列出所有知识库分组的名称。

**Returns:**
- list: 知识库分组名称列表。
""")

add_chinese_doc('rag.utils.DocListManager.add_files', '''\
批量向文档列表中添加文件，可选附加元数据、状态，并支持分批处理。
此方法将文件列表添加到数据库中，并为每个文件设置可选的元数据和初始状态。文件会以批量方式处理以提高效率。在文件添加完成后，它们会自动关联到默认的知识库 (KB) 组。
Args:
    files (List[str]): 添加的文件路径列表。
    metadatas (Optional[List[Dict[str, Any]]]): 与文件对应的元数据字典列表。默认为 `None`。
    status (Optional[str]): 添加文件的初始状态。默认为 `Status.waiting`。
    batch_size (int): 每批处理的文件数量。默认为 64。
**Returns:**:
    List[DocPartRow]: 包含已添加文件及其相关信息的 `DocPartRow` 对象列表。
说明:
    - 方法首先通过辅助函数 `_add_doc_records` 创建文档记录。
    - 文件添加后，会自动关联到默认的知识库组 (`DocListManager.DEFAULT_GROUP_NAME`)。
    - 批量处理确保在添加大量文件时具有良好的可扩展性。
''')

add_chinese_doc('rag.utils.DocListManager.delete_unreferenced_doc', '''\
删除数据库中标记为 "删除中" 且不再被引用的文档。
此方法从数据库中删除满足以下条件的文档：
1. 文档状态为 `DocListManager.Status.deleting`。
2. 文档的引用计数 (`count`) 为 0。
''')

add_chinese_doc('rag.utils.DocListManager.get_docs_need_reparse', '''\
获取需要重新解析 (`need_reparse=True`)的指定组中的文档。
此方法检索标记为需要重新解析 (`need_reparse=True`) 的文档，基于提供的组名。仅包含状态为 `success` 或 `failed` 的文档。
Args:
    group (str): 用于过滤文档的组名。
**Returns:**:
    List[KBDocument]: 需要重新解析的 `KBDocument` 对象列表。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - 查询通过 SQL `JOIN` 操作连接 `KBDocument` 和 `KBGroupDocuments` 表，并基于组名和重新解析状态进行过滤。
    - 仅状态为 `success` 或 `failed` 且 `need_reparse=True` 的文档会被检索出来。
''')

add_chinese_doc('rag.utils.DocListManager.get_existing_paths_by_pattern', '''\
根据给定的模式，检索符合条件的文档路径。
此方法从数据库中获取所有符合提供的 SQL `LIKE` 模式的文档路径。
Args:
    pattern (str): 用于过滤文档路径的 SQL `LIKE` 模式。例如，`%example%` 匹配包含单词 "example" 的路径。
**Returns:**:
    List[str]: 符合给定模式的文档路径列表。如果没有匹配的路径，则返回空列表。
说明:
    - 使用线程安全锁 (`self._db_lock`) 确保多线程环境下的数据库访问安全。
    - SQL 查询中的 `LIKE` 操作符用于对文档路径进行模式匹配。
''')

add_chinese_doc('rag.utils.DocListManager.update_file_message', """\
更新指定文件的消息。

Args:
    fileid (str): 文件ID。
    **kw: 需要更新的其他键值对。
""")

add_chinese_doc('rag.utils.DocListManager.update_file_status', """\
更新指定文件的状态。

Args:
    file_ids (list of str): 更新状态的文件ID列表。
    status (str): 目标状态。
    cond_status_list(Union[None, List[str]]):限制只更新处于这些状态的文档
""")

add_chinese_doc('rag.utils.DocListManager.add_files_to_kb_group', """\
将文件添加到指定的知识库分组中。

Args:
    file_ids (list of str): 要添加的文件ID列表。
    group (str): 要添加的分组名称。
""")

add_chinese_doc('rag.utils.DocListManager.delete_files_from_kb_group', """\
从指定的知识库分组中删除文件。

Args:
    file_ids (list of str): 要删除的文件ID列表。
    group (str): 分组名称。
""")

add_chinese_doc('rag.utils.DocListManager.get_file_status', """\
获取指定文件的状态。

Args:
    fileid (str): 文件ID。

**Returns:**
- str: 文件的当前状态。
""")

add_chinese_doc('rag.utils.DocListManager.update_kb_group', """\
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

add_chinese_doc('rag.utils.DocListManager.release', """\
释放当前管理器的资源。

""")

add_chinese_doc('rag.utils.DocListManager.enable_path_monitoring', '''\
启用或禁用文档管理器的路径监控功能。
此方法用于启用或禁用文档管理器的路径监控功能。当启用时，会启动一个监控线程处理与路径相关的操作；当禁用时，会停止该线程并等待它终止。
Args:
    val (bool): 启用或禁用路径监控。
说明:
    - 如果 `val` 为 `True`，路径监控功能会通过将 `_monitor_continue` 设置为 `True` 并启动 `_monitor_thread` 来启用。
    - 如果 `val` 为 `False`，路径监控功能会通过将 `_monitor_continue` 设置为 `False` 并等待 `_monitor_thread` 终止来禁用。
    - 方法在管理监控线程时确保线程操作是安全的。
''')

add_english_doc('rag.utils.DocListManager', """\
Abstract base class for managing document lists and monitoring changes in a document directory.

Args:
    path: Path of the document directory to monitor.
    name: Name of the manager.
    enable_path_monitoring: Whether to enable path monitoring.
""")

add_english_doc('rag.utils.DocListManager.init_tables', """\
Ensure that the default group exists in the database tables.
""")

add_english_doc('rag.utils.DocListManager.delete_files', """\
Set the knowledge base entries associated with the document to "deleting," and have each knowledge base asynchronously delete parsed results and associated records.

Args:
    file_ids (list of str): List of file IDs to delete.
""")

add_english_doc('rag.utils.DocListManager.table_inited', """\
Checks if the database table `documents` is initialized. This method ensures thread-safety when accessing the database.
Determines whether the `documents` table exists in the database.
Returns:
    bool: `True` if the `documents` table exists, `False` otherwise.
Notes:
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe access to the database.
    - Establishes a connection to the SQLite database at `self._db_path` with the `check_same_thread` option.
    - Executes the SQL query: `SELECT name FROM sqlite_master WHERE type='table' AND name='documents'` to check for the table.
""")

add_english_doc('rag.utils.DocListManager.validate_paths', '''\
Validates a list of file paths to ensure they are ready for processing.
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


add_english_doc('rag.utils.DocListManager.update_need_reparsing', '''\
Updates the `need_reparse` status of a document in the `KBGroupDocuments` table.
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

add_english_doc('rag.utils.DocListManager.list_files', """\
Lists files from the `documents` table with optional filtering, limiting, and returning details.
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
""")

add_english_doc('rag.utils.DocListManager.get_docs', '''\
This method retrieves document objects of type `KBDocument` from the database for the provided list of document IDs.
Args:
    doc_ids (List[str]): A list of document IDs to fetch.
Returns:
    List[KBDocument]: A list of `KBDocument` objects corresponding to the provided document IDs. If no documents are found, an empty list is returned.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - It performs a SQL join between `KBDocument` and `KBGroupDocuments` to retrieve the relevant rows.
    - After fetching, it updates the `new_meta` field of the affected rows to `None` and commits the changes to the database.
''')

add_english_doc('rag.utils.DocListManager.set_docs_new_meta', """\
Batch update metadata for documents.

Args:
    doc_meta (Dict[str, dict]): A dictionary mapping document IDs to their new metadata.
""")

add_english_doc('rag.utils.DocListManager.fetch_docs_changed_meta', '''\
List files in a specific knowledge base (KB) group with optional filters, limiting, and details.
This method retrieves files from the `kb_group_documents` table, optionally filtering by group, document status, upload status, and whether reparsing is needed.
Args:
    group (str): The name of the group to filter documents by.
**Returns:**
    List[DocMetaChangedRow]: A list of rows, where each row contains the `doc_id` and the `new_meta` field of documents with changed metadata.
Notes:
    - This method constructs a SQL query dynamically based on the provided filters.
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - If `status` or `upload_status` are provided as lists, they are processed with SQL `IN` clauses.
''')

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

add_english_doc('rag.utils.DocListManager.list_kb_group_files', '''\
List files in a specific knowledge base group .

Args:
    group (str): The name of the KB group to filter files by. Defaults to `None` .
    limit (Optional[int]): Maximum number of files to return. If `None`, returns all matching files.
    details (bool): Whether to return detailed file information or only file IDs and paths.
    status (Union[str, List[str]]): The KB group status or list of statuses to include in the results. Defaults to all statuses.
    exclude_status (Optional[Union[str, List[str]]): The KB group status or list of statuses to exclude from the results. Defaults to `None`.
    upload_status (Union[str, List[str]]): The document upload status or list of statuses to include in the results. Defaults to all statuses.
    exclude_upload_status (Optional[Union[str, List[str]]): The document upload status or list of statuses to exclude from the results. Defaults to `None`.
    need_reparse (Optional[bool]): Whether to filter files that need reparsing or not . Defaults to `None` .
**Returns:**:
    List: If `details=False`, returns a list of tuples containing `(doc_id, path)`. 
          If `details=True`, returns a list of detailed rows with additional metadata.
Notes:
    - The method first creates document records using the `_add_doc_records` helper function.
    - After the files are added, they are automatically linked to the default KB group (`DocListManager.DEFAULT_GROUP_NAME`).
    - Batch processing ensures scalability when adding a large number of files.
''')

add_english_doc('rag.utils.DocListManager.add_files', '''\
Add multiple files to the document list with optional metadata, status, and batch processing.
This method adds a list of files to the database and sets optional metadata and initial status for each file. The files are processed in batches for efficiency. After the files are added, they are automatically associated with the default knowledge base (KB) group.
Args:
    files (List[str]): A list of file paths to add to the database.
    metadatas (Optional[List[Dict[str, Any]]]): A list of metadata dictionaries corresponding to the files. If `None`, no metadata will be associated. Defaults to `None`.
    status (Optional[str]): The initial status for the added files. Defaults to `Status.waiting`.
    batch_size (int): The number of files to process in each batch. Defaults to 64.
**Returns:**:
    List[DocPartRow]: A list of `DocPartRow` objects representing the added files and their associated information.
Notes:
- The method first creates document records using the helper function _add_doc_records.
- After the files are added, they are automatically linked to the default knowledge base group (DocListManager.DEFAULT_GROUP_NAME).
- Batch processing ensures good scalability when adding a large number of files.


''')

add_english_doc('rag.utils.DocListManager.delete_unreferenced_doc', '''\
Delete documents marked as "deleting" and no longer referenced in the database.
This method removes documents from the database that meet the following conditions:
1. Their status is set to `DocListManager.Status.deleting`.
2. Their reference count (`count`) is 0.
''')

add_english_doc('rag.utils.DocListManager.get_docs_need_reparse', '''\
Retrieve documents that require reparsing for a specific group.
This method fetches documents that are marked as needing reparsing (`need_reparse=True`) for the given group. Only documents with a status of `success` or `failed` are included in the results.
Args:
    group (str): The name of the group to filter documents by.
**Returns:**:
    List[KBDocument]: A list of `KBDocument` objects that need reparsing.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The query performs a SQL `JOIN` between `KBDocument` and `KBGroupDocuments` to filter by group and reparse status.
    - Documents with `need_reparse=True` and a status of `success` or `failed` are considered for reparsing.
''')

add_english_doc('rag.utils.DocListManager.get_existing_paths_by_pattern', '''\
Retrieve existing document paths that match a given pattern.
This method fetches all document paths from the database that match the provided SQL `LIKE` pattern.
Args:
    pattern (str): The SQL `LIKE` pattern to filter document paths. For example, `%example%` matches paths containing the word "example".
**Returns:**:
    List[str]: A list of document paths that match the given pattern. If no paths match, an empty list is returned.
Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The `LIKE` operator in the SQL query is used to perform pattern matching on document paths.
''')

add_english_doc('rag.DocListManager.update_file_message', """\
Updates the message for a specified file.

Args:
    fileid (str): File ID.
    **kw: Additional key-value pairs to update.
""")

add_english_doc('rag.DocListManager.update_file_status', """\
Update the status of specified files.

Args:
    file_ids (list of str): List of file IDs whose status needs to be updated.
    status (str): Target status to set.
    cond_status_list (Union[None, List[str]]): Optional. Only update files currently in these statuses.
""")

add_english_doc('rag.DocListManager.add_files_to_kb_group', """\
Adds files to the specified knowledge base group.

Args:
    file_ids (list of str): List of file IDs to add.
    group (str): Name of the group to add the files to.
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

add_english_doc('rag.utils.DocListManager.enable_path_monitoring', '''\
Enable or disable path monitoring for the document manager.
This method enables or disables the path monitoring functionality in the document manager. When enabled, a monitoring thread starts to handle path-related operations. When disabled, the thread stops and joins (waits for it to terminate).
Args:
    val (bool): Whether to enable or disable path monitoring.
Notes:
    - If `val` is `True`, path monitoring is enabled by setting `_monitor_continue` to `True` and starting the `_monitor_thread`.
    - If `val` is `False`, path monitoring is disabled by setting `_monitor_continue` to `False` and joining the `_monitor_thread` if it is running.
    - This method ensures thread-safe operation when managing the monitoring thread.
''')

add_example('rag.utils.DocListManager', '''
>>> import lazyllm
>>> from lazyllm.rag.utils import DocListManager
>>> manager = DocListManager(path='your_file_path/', name="test_manager", enable_path_monitoring=False)
>>> added_docs = manager.add_files([test_file_list])
>>> manager.enable_path_monitoring(True)
>>> deleted = manager.delete_files([delete_file_list])
''')

add_chinese_doc('rag.utils.SqliteDocListManager', '''\
基于 SQLite 的文档管理器，用于本地文件的持久化存储、状态管理与元信息追踪。

该类继承自 DocListManager，利用 SQLite 数据库存储文档记录。适用于管理具有唯一标识符的本地文档资源，并提供便捷的插入、查询、更新与状态过滤接口，支持可选的路径监控功能。

Args:
    path (str): 数据库存储路径。
    name (str): 数据库文件名（不包含路径）。
    enable_path_monitoring (bool): 是否启用对文件路径的变动监控，默认为 True。
''')

add_english_doc('rag.utils.SqliteDocListManager', '''\
SQLite-based document manager for persistent local file storage, status tracking, and metadata management.

This class inherits from DocListManager and uses a SQLite backend to store document records. It is suitable for managing locally identified documents with support for inserting, querying, updating, and filtering based on status. Optional file path monitoring is also supported.

Args:
    path (str): Directory path to store the database.
    name (str): Name of the SQLite database file (without path).
    enable_path_monitoring (bool): Whether to enable path monitoring. Defaults to True.
''')

add_example('rag.utils.SqliteDocListManager', '''\
>>> from lazyllm.tools.rag.utils import SqliteDocListManager
>>> manager = SqliteDocListManager(path="./data", name="docs.sqlite")
>>> manager.insert({"uid": "doc_001", "name": "example.txt", "status": "ready"})
>>> print(manager.get("doc_001"))
>>> files = manager.list_files(limit=5, details=True)
>>> print(files)
''')

add_chinese_doc('rag.utils.SqliteDocListManager.table_inited', '''\
检查数据库中是否已存在名为 "documents" 的表。

该方法通过查询 sqlite_master 元信息表，判断数据表是否已初始化。

**Returns:**\n
- bool: 如果 "documents" 表存在，返回 True；否则返回 False。
''')

add_english_doc('rag.utils.SqliteDocListManager.table_inited', '''\
Checks whether the "documents" table has been initialized in the database.

The method queries the sqlite_master metadata table to verify if the "documents" table exists.

**Returns:**\n
- bool: True if the "documents" table exists, False otherwise.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.get_status_cond_and_params', '''\
生成用于文档状态筛选的 SQL 条件语句及其参数列表。

根据传入的包含状态和排除状态，构造 WHERE 子句中使用的 SQL 表达式。支持字段名前缀，用于联表查询等场景。

Args:
    status (str 或 list of str): 要包含的文档状态。若为 "all"，不添加包含条件。
    exclude_status (str 或 list of str, optional): 要排除的文档状态。不能为 "all"。
    prefix (str, optional): 字段名前缀（如联表查询中的别名），将应用于字段名。

**Returns:**\n
- Tuple[str, list]: 包含 SQL 条件语句和对应参数的元组。
''')


add_english_doc('rag.utils.SqliteDocListManager.get_status_cond_and_params', '''\
Generates SQL condition expressions and parameter values for filtering documents by status.

Builds WHERE clause components using the given inclusion and exclusion statuses. Supports field name prefixing for use in joined queries.

Args:
    status (str or list of str): Document status(es) to include. If set to "all", no inclusion condition will be applied.
    exclude_status (str or list of str, optional): Status(es) to exclude. Must not be "all".
    prefix (str, optional): Optional field prefix (e.g., table alias) to prepend to the status field.

**Returns:**\n
- Tuple[str, list]: A tuple containing the SQL condition string and its corresponding parameter values.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.validate_paths', '''\
验证输入路径所对应的文档是否可以安全添加到数据库。

该方法会检查每个路径是否对应已有文档，若已存在，需判断其状态是否允许重解析。
若文档正在解析或等待解析，或上次重解析未完成，则视为不可用。

Args:
    paths (List[str]): 文件路径列表。

**Returns:**\n
- Tuple[bool, str, List[bool]]: 
    - bool: 是否所有路径都验证通过。
    - str: 成功或失败的描述信息。
    - List[bool]: 与输入路径一一对应的布尔列表，表示该路径是否为新文档（True 为新文档，False 为已存在）。
        若验证失败，返回值为 None。
''')

add_english_doc('rag.utils.SqliteDocListManager.validate_paths', '''\
Validates whether the documents corresponding to the given paths can be safely added to the database.

The method checks if the document already exists. If it exists, it verifies whether the document is currently
being parsed, waiting to be parsed, or was not successfully re-parsed last time.

Args:
    paths (List[str]): A list of file paths to validate.

**Returns:**\n
- Tuple[bool, str, List[bool]]: 
    - bool: Whether all paths passed validation.
    - str: Description message of the validation result.
    - List[bool]: A boolean list corresponding to input paths, indicating whether each path is new (True) or already exists (False).
      If validation fails, this value is None.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.update_need_reparsing', '''\
更新指定文档的重解析标志位。

该方法用于设置某个文档是否需要重新解析。可以选择性地指定知识库分组进行精确匹配。

Args:
    doc_id (str): 文档的唯一标识符。
    need_reparse (bool): 是否需要重新解析文档。
    group_name (Optional[str]): 可选，所属的知识库分组名称。如果提供，将仅更新指定分组中的文档。
''')

add_english_doc('rag.utils.SqliteDocListManager.update_need_reparsing', '''\
Updates the re-parsing flag for a specific document.

This method sets whether a document should be re-parsed. If a group name is provided, the update is scoped to that group only.

Args:
    doc_id (str): The unique identifier of the document.
    need_reparse (bool): Whether the document needs to be re-parsed.
    group_name (Optional[str]): Optional. The knowledge base group name to filter by. If provided, only documents in the specified group will be updated.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.list_files', """\
列出文档数据库中符合状态条件的文件，并根据参数选择返回完整记录或仅返回文件路径。

Args:
    limit (Optional[int]): 要返回的记录数上限，若为 None 则返回所有符合条件的记录。
    details (bool): 是否返回完整的数据库行信息，若为 False 则仅返回文档路径（ID）。
    status (Union[str, List[str]]): 要包含在结果中的状态值，默认为包含所有状态。
    exclude_status (Optional[Union[str, List[str]]]): 要从结果中排除的状态值。

**Returns:**\n
- list: 文件记录列表或文档路径列表，具体取决于 `details` 参数。
""")

add_english_doc('rag.utils.SqliteDocListManager.list_files', """\
Lists files in the document database based on status filters and returns either full records or file paths.

Args:
    limit (Optional[int]): The maximum number of records to return. If None, all matching records are returned.
    details (bool): Whether to return full database rows or just file paths (document IDs).
    status (Union[str, List[str]]): Status values to include in the result. Defaults to including all.
    exclude_status (Optional[Union[str, List[str]]]): Status values to exclude from the result.

**Returns:**\n
- list: A list of file records or document paths depending on the `details` flag.
""")

add_chinese_doc('rag.utils.SqliteDocListManager.get_docs', '''\
根据给定的文档ID列表，从数据库中获取对应的文档对象列表。

Args:
    doc_ids (List[str]): 需要查询的文档ID列表。

**Returns:**\n
- List[KBDocument]: 匹配的文档对象列表。如果没有匹配项，返回空列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.get_docs', '''\
Fetches document objects from the database corresponding to the given list of document IDs.

Args:
    doc_ids (List[str]): A list of document IDs to query.

**Returns:**\n
- List[KBDocument]: A list of matching document objects. Returns an empty list if no matches found.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.set_docs_new_meta', '''\
批量更新文档的元数据（meta），同时更新对应知识库分组中文档的 new_meta 字段（非等待状态的文档）。

Args:
    doc_meta (Dict[str, dict]): 字典，键为文档ID，值为对应的新元数据字典。
''')

add_english_doc('rag.utils.SqliteDocListManager.set_docs_new_meta', '''\
Batch updates the metadata (meta) of documents, and simultaneously updates the new_meta field of documents in knowledge base groups for documents that are not in waiting status.

Args:
    doc_meta (Dict[str, dict]): A dictionary mapping document IDs to their new metadata dictionaries.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.fetch_docs_changed_meta', '''\
获取指定知识库分组中元数据发生变化的文档列表，并将对应的 new_meta 字段清空。

Args:
    group (str): 知识库分组名称。

**Returns:**\n
- List[DocMetaChangedRow]: 包含文档ID及其对应新元数据的列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.fetch_docs_changed_meta', '''\
Fetches the list of documents within a specified knowledge base group that have updated metadata, and resets the new_meta field for those documents.

Args:
    group (str): Name of the knowledge base group.

**Returns:**\n
- List[DocMetaChangedRow]: A list containing document IDs and their updated metadata.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.list_all_kb_group', '''\
列出数据库中所有的知识库分组名称。

**Returns:**\n
- List[str]: 知识库分组名称列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.list_all_kb_group', '''\
Lists all knowledge base group names stored in the database.

**Returns:**\n
- List[str]: A list of knowledge base group names.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.add_kb_group', '''\
向数据库中添加一个新的知识库分组名称，若已存在则忽略。

Args:
    name (str): 要添加的知识库分组名称。
''')

add_english_doc('rag.utils.SqliteDocListManager.add_kb_group', '''\
Adds a new knowledge base group name to the database; ignores if the group already exists.

Args:
    name (str): The name of the knowledge base group to add.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.list_kb_group_files', '''\
列出指定知识库分组中的文件信息，可根据多种条件进行过滤。

Args:
    group (str, optional): 知识库分组名称，若为 None 则不按分组过滤。
    limit (int, optional): 限制返回的文件数量。
    details (bool): 是否返回详细的文件信息。
    status (str or List[str], optional): 过滤知识库分组中文件的状态。
    exclude_status (str or List[str], optional): 排除指定状态的文件。
    upload_status (str or List[str], optional): 过滤文件上传状态。
    exclude_upload_status (str or List[str], optional): 排除指定的上传状态。
    need_reparse (bool, optional): 是否只返回需要重新解析的文件。

**Returns:**\n
- list: 
    - 如果 details 为 False，返回列表，每个元素为 (doc_id, path) 元组。
    - 如果 details 为 True，返回包含文件详细信息的元组列表，包括文档ID、路径、状态、元数据，
      知识库分组名、分组内状态及日志。
''')

add_english_doc('rag.utils.SqliteDocListManager.list_kb_group_files', '''\
Lists files in a specified knowledge base group, with support for multiple filters.

Args:
    group (str, optional): Knowledge base group name to filter by. If None, no group filtering is applied.
    limit (int, optional): Limit on the number of files to return.
    details (bool): Whether to return detailed file information.
    status (str or List[str], optional): Filter files by group document status.
    exclude_status (str or List[str], optional): Exclude files with these group document statuses.
    upload_status (str or List[str], optional): Filter files by upload document status.
    exclude_upload_status (str or List[str], optional): Exclude files with these upload document statuses.
    need_reparse (bool, optional): If set, only returns files marked as needing reparse.

**Returns:**\n
- list: 
    - If details is False, returns a list of tuples (doc_id, path).
    - If details is True, returns a list of tuples containing detailed file information:
      document ID, path, status, metadata, group name, group status, and group log.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.delete_unreferenced_doc', '''\
删除数据库中标记为删除且未被任何知识库分组引用的文档记录。

该方法会查找状态为“deleting”且引用计数为0的文档，删除这些文档记录，并记录删除操作日志。

''')

add_english_doc('rag.utils.SqliteDocListManager.delete_unreferenced_doc', '''\
Deletes documents from the database that are marked for deletion and are no longer referenced by any knowledge base group.

This method queries documents with status "deleting" and a reference count of zero, deletes them from the database,
and adds operation logs for these deletions.

''')

add_chinese_doc('rag.utils.SqliteDocListManager.get_docs_need_reparse', '''\
获取指定知识库分组中需要重新解析的文档列表。

仅返回状态为“success”或“failed”的文档，且其对应的知识库分组记录标记为需要重新解析。

Args:
    group (str): 知识库分组名称。

**Returns:**\n
- List[KBDocument]: 需要重新解析的文档列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.get_docs_need_reparse', '''\
Retrieves the list of documents that require re-parsing within a specified knowledge base group.

Only documents with status "success" or "failed" and marked as needing reparse in the group are returned.

Args:
    group (str): Name of the knowledge base group.

**Returns:**\n
- List[KBDocument]: List of documents that need to be re-parsed.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.get_existing_paths_by_pattern', '''\
根据路径匹配模式获取已存在的文档路径列表。

Args:
    pattern (str): 路径匹配模式，支持SQL的LIKE通配符。

**Returns:**\n
- List[str]: 匹配到的已存在文档路径列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.get_existing_paths_by_pattern', '''\
Retrieves a list of existing document paths that match a given pattern.

Args:
    pattern (str): Path matching pattern, supports SQL LIKE wildcards.

**Returns:**\n
- List[str]: List of existing document paths matching the pattern.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.update_file_message', '''\
更新指定文件的字段信息。

Args:
    fileid (str): 文件的唯一标识符（doc_id）。
    **kw: 需要更新的字段及其对应的值，键值对形式传入。
''')

add_english_doc('rag.utils.SqliteDocListManager.update_file_message', '''\
Updates fields of the specified file record.

Args:
    fileid (str): Unique identifier of the file (doc_id).
    **kw: Key-value pairs of fields to update and their new values.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.update_file_status', '''\
更新多个文件的状态，支持根据当前状态进行条件过滤。

Args:
    file_ids (List[str]): 需要更新状态的文件ID列表。
    status (str): 要设置的新状态。
    cond_status_list (Union[None, List[str]], optional): 仅更新当前状态在此列表中的文件，默认为 None，表示不筛选。

**Returns:**\n
- List[DocPartRow]: 返回更新后的文件ID和路径列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.update_file_status', '''\
Updates the status of multiple files, optionally filtered by current status.

Args:
    file_ids (List[str]): List of file IDs to update.
    status (str): New status to set.
    cond_status_list (Union[None, List[str]], optional): List of statuses to filter files that can be updated. Defaults to None.

**Returns:**\n
- List[DocPartRow]: List of updated file IDs and their paths.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.add_files_to_kb_group', '''\
将多个文件添加到指定的知识库分组中。

该方法会将文件状态设置为等待处理（waiting），
若添加成功，则对应文档的计数（count）加一。

Args:
    file_ids (List[str]): 需要添加的文件ID列表。
    group (str): 知识库分组名称。
''')

add_english_doc('rag.utils.SqliteDocListManager.add_files_to_kb_group', '''\
Adds multiple files to the specified knowledge base group.

This method sets the file status to waiting.
If successfully added, increments the document's count.

Args:
    file_ids (List[str]): List of file IDs to add.
    group (str): Name of the knowledge base group.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.delete_files_from_kb_group', '''\
从指定的知识库分组中删除多个文件。

删除成功后，对应文档的计数（count）减少，但不会低于0。
若文档不存在，会记录警告日志。

Args:
    file_ids (List[str]): 需要删除的文件ID列表。
    group (str): 知识库分组名称。
''')

add_english_doc('rag.utils.SqliteDocListManager.delete_files_from_kb_group', '''\
Deletes multiple files from the specified knowledge base group.

After deletion, decrements the document's count but not below zero.
If the document is not found, logs a warning.

Args:
    file_ids (List[str]): List of file IDs to delete.
    group (str): Name of the knowledge base group.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.get_file_status', '''\
获取指定文件的状态。

Args:
    fileid (str): 文件的唯一标识符。

**Returns:**\n
- Optional[Tuple]: 返回包含状态的元组，若文件不存在则返回 None。
''')

add_english_doc('rag.utils.SqliteDocListManager.get_file_status', '''\
Gets the status of a specified file.

Args:
    fileid (str): Unique identifier of the file.

**Returns:**\n
- Optional[Tuple]: A tuple containing the status, or None if the file does not exist.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.update_kb_group', '''\
更新知识库分组中指定文件的状态和重解析需求。

根据给定的文件ID列表、分组名及状态列表，批量更新对应文件在知识库分组中的状态及是否需要重解析标志。

Args:
    cond_file_ids (List[str]): 需要更新的文件ID列表。
    cond_group (Optional[str]): 分组名称，若指定则只更新该分组内的文件。
    cond_status_list (Optional[List[str]]): 仅更新状态匹配此列表的文件。
    new_status (Optional[str]): 新的文件状态。
    new_need_reparse (Optional[bool]): 新的重解析需求标志。

**Returns:**\n
- List[Tuple]: 返回更新后文件的doc_id、group_name及状态列表。
''')

add_english_doc('rag.utils.SqliteDocListManager.update_kb_group', '''\
Updates the status and reparse need flag of specified files in a knowledge base group.

Batch updates files' status and need_reparse flag within a knowledge base group based on file IDs, group name, and optional status filter.

Args:
    cond_file_ids (List[str]): List of file IDs to update.
    cond_group (Optional[str]): Group name to filter files, if specified only updates files in this group.
    cond_status_list (Optional[List[str]]): Only update files whose status is in this list.
    new_status (Optional[str]): New status to set.
    new_need_reparse (Optional[bool]): New flag indicating if reparse is needed.

**Returns:**\n
- List[Tuple]: List of tuples of updated files containing doc_id, group_name, and status.
''')

add_chinese_doc('rag.utils.SqliteDocListManager.release', '''\
清空数据库中的所有文档、分组及相关操作日志数据。

该操作会删除 documents、document_groups、kb_group_documents 和 operation_logs 表中的所有记录。

''')

add_english_doc('rag.utils.SqliteDocListManager.release', '''\
Clears all documents, groups, and operation logs from the database.

This operation deletes all records from documents, document_groups, kb_group_documents, and operation_logs tables.

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

add_chinese_doc('CodeGenerator.choose_prompt', '''\
根据输入的提示文本内容选择合适的代码生成提示模板。  
如果提示中包含中文字符，则返回中文提示模板；否则返回英文提示模板。

Args:
    prompt (str): 输入的提示文本。

**Returns:**\n
- str: 选择的代码生成提示模板字符串。
''')

add_english_doc('CodeGenerator.choose_prompt', '''\
Selects an appropriate code generation prompt template based on the content of the input prompt.  
Returns the Chinese prompt template if Chinese characters are detected; otherwise returns the English prompt template.

Args:
    prompt (str): Input prompt text.

**Returns:**\n
- str: The selected code generation prompt template string.
''')

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

# QustionRewrite.choose_prompt
add_english_doc('QustionRewrite.choose_prompt', '''
Choose the appropriate prompt template based on the language of the input prompt.

This method analyzes the input prompt string and determines whether to use the Chinese or English prompt template. It checks each character in the prompt string and if any character falls within the Chinese Unicode range (\\u4e00-\\u9fff), it returns the Chinese prompt template; otherwise, it returns the English prompt template.

Args:
    prompt (str): The input prompt string to be analyzed for language detection.

Returns:
    str: The selected prompt template string (either Chinese or English version).
''')

add_chinese_doc('QustionRewrite.choose_prompt', '''
根据输入提示的语言选择合适的提示模板。

此方法分析输入提示字符串并确定使用中文还是英文提示模板。它检查提示字符串中的每个字符，如果任何字符落在中文字符Unicode范围内（\\u4e00-\\u9fff），则返回中文提示模板；否则返回英文提示模板。

Args:
    prompt (str): 要分析语言检测的输入提示字符串。

Returns:
    str: 选定的提示模板字符串（中文或英文版本）。
''')

add_example('QustionRewrite.choose_prompt', '''
>>> from lazyllm.tools.actors.qustion_rewrite import QustionRewrite

# Example 1: English prompt (no Chinese characters)
>>> rewriter = QustionRewrite("gpt-3.5-turbo")
>>> prompt_template = rewriter.choose_prompt("How to implement machine learning?")
>>> print("Template contains Chinese:", "中文" in prompt_template)
Template contains Chinese: False

# Example 2: Chinese prompt (contains Chinese characters)
>>> prompt_template = rewriter.choose_prompt("如何实现机器学习？")
>>> print("Template contains Chinese:", "中文" in prompt_template)
Template contains Chinese: True

# Example 3: Mixed language prompt (contains Chinese characters)
>>> prompt_template = rewriter.choose_prompt("What is 机器学习?")
>>> print("Template contains Chinese:", "中文" in prompt_template)
Template contains Chinese: True
''')

add_chinese_doc('ToolManager', '''\
ToolManager是一个工具管理类，用于提供工具信息和工具调用给function call。

此管理类构造时需要传入工具名字符串列表。此处工具名可以是LazyLLM提供的，也可以是用户自定义的，如果是用户自定义的，首先需要注册进LazyLLM中才可以使用。在注册时直接使用 `fc_register` 注册器，该注册器已经建立 `tool` group，所以使用该工具管理类时，所有函数都统一注册进 `tool` 分组即可。待注册的函数需要对函数参数进行注解，并且需要对函数增加功能描述，以及参数类型和作用描述。以方便工具管理类能对函数解析传给LLM使用。

Args:
    tools (List[str]): 工具名称字符串列表。
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

add_chinese_doc("ModuleTool.apply", '''
抽象方法，需在子类中实现具体逻辑。

此方法应根据传入的参数执行特定任务。

Raises:
    NotImplementedError: 如果未在子类中重写该方法。
''')

add_english_doc("ModuleTool.apply", '''
Abstract method to be implemented in subclasses.

This method should perform a specific task based on the provided arguments.

Raises:
    NotImplementedError: If the method is not overridden in a subclass.
''')

add_chinese_doc("ModuleTool.validate_parameters", '''
验证参数是否满足所需条件。

此方法会检查参数字典是否包含所有必须字段，并尝试进一步进行格式验证。

Args:
    arguments (Dict[str, Any]): 传入的参数字典。

Returns:
    bool: 若参数合法且完整，返回 True；否则返回 False。
''')

add_english_doc("ModuleTool.validate_parameters", '''
Validate whether the provided arguments meet the required criteria.

This method checks if all required keys are present in the input dictionary and attempts format validation.

Args:
    arguments (Dict[str, Any]): Dictionary of input arguments.

Returns:
    bool: True if valid and complete; False otherwise.
''')

add_chinese_doc('FunctionCall', '''\
FunctionCall是单轮工具调用类。当LLM自身信息不足以回答用户问题，需要结合外部工具获取辅助信息时，调用此类。  
若LLM输出需要调用工具，则执行工具调用并返回调用结果；输出结果为List类型，包含当前轮的输入、模型输出和工具输出。  
若不需工具调用，则直接返回LLM输出结果，输出为字符串类型。

Args:
    llm (ModuleBase): 使用的LLM实例，支持TrainableModule或OnlineChatModule。
    tools (List[Union[str, Callable]]): LLM可调用的工具名称或Callable对象列表。
    return_trace (Optional[bool]): 是否返回调用轨迹，默认为False。
    stream (Optional[bool]): 是否启用流式输出，默认为False。
    _prompt (Optional[str]): 自定义工具调用提示语，默认根据llm类型自动设置。

注意：tools中的工具需包含`__doc__`字段，且须遵循[Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)规范说明用途与参数。
''')

add_english_doc('FunctionCall', '''\
FunctionCall is a single-turn tool invocation class. It is used when the LLM alone cannot answer user queries and requires external knowledge through tool calls.  
If the LLM output requires tool calls, the tools are invoked and the combined results (input, model output, tool output) are returned as a list.  
If no tool calls are needed, the LLM output is returned directly as a string.

Args:
    llm (ModuleBase): The LLM instance to use, which can be either a TrainableModule or OnlineChatModule.
    tools (List[Union[str, Callable]]): A list of tool names or callable objects that the LLM can use.
    return_trace (Optional[bool]): Whether to return the invocation trace, defaults to False.
    stream (Optional[bool]): Whether to enable streaming output, defaults to False.
    _prompt (Optional[str]): Custom prompt for function call, defaults to automatic selection based on llm type.

Note: Tools in `tools` must include a `__doc__` attribute and describe their purpose and parameters according to the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
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

add_chinese_doc('ReactAgent', '''\
ReactAgent是按照 `Thought->Action->Observation->Thought...->Finish` 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM 使用的工具名称列表。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
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

add_chinese_doc('DBManager.execute_query', '''\
执行数据库查询语句的抽象方法。此方法需要由具体的数据库管理器子类实现，用于执行各种数据库操作。

Args:
    statement: 要执行的数据库查询语句，可以是 SQL 语句或其他数据库特定的查询语言

此方法的特点：

- **抽象方法**: 需要在子类中实现具体的数据库操作逻辑
- **统一接口**: 为不同的数据库类型提供统一的查询接口
- **错误处理**: 子类实现应该包含适当的错误处理和状态报告
- **结果格式化**: 返回格式化的字符串结果，便于后续处理

**注意**: 此方法是数据库管理器的核心方法，所有具体的数据库操作都通过此方法执行。

''')

add_english_doc('DBManager.execute_query', '''\
Abstract method for executing database query statements. This method needs to be implemented by specific database manager subclasses to execute various database operations.

Args:
    statement: The database query statement to execute, which can be SQL statements or other database-specific query languages

Features of this method:

- **Abstract Method**: Requires implementation of specific database operation logic in subclasses
- **Unified Interface**: Provides a unified query interface for different database types
- **Error Handling**: Subclass implementations should include appropriate error handling and status reporting
- **Result Formatting**: Returns formatted string results for subsequent processing

**Note**: This method is the core method of the database manager, and all specific database operations are executed through this method.

''')

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
    >>> sql_llm = lazyllm.OnlineChatModule(model="gpt-4o", source="openai", base_url="***")
    >>> sql_call = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=True)
    >>> print(sql_call("去年一整年销售额最多的员工是谁?"))
""",
)

add_english_doc('SqlCall.sql_query_promt_hook', '''\
Hook to prepare the prompt inputs for generating a database query from user input.

Args:
    input (Union[str, List, Dict[str, str], None]): The user's natural language query.
    history (List[Union[List[str], Dict[str, Any]]]): Conversation history.
    tools (Union[List[Dict[str, Any]], None]): Available tool descriptions.
    label (Union[str, None]): Optional label for the prompt.

Returns:
    Tuple: A tuple containing the formatted prompt dict (with current_date, db_type, desc, user_query), history, tools, and label.
''')

add_chinese_doc('SqlCall.sql_query_promt_hook', r'''\ 
为从用户输入生成数据库查询准备 prompt 的 hook。

Args:
    input (Union[str, List, Dict[str, str], None]): 用户的自然语言查询。
    history (List[Union[List[str], Dict[str, Any]]]): 会话历史。
    tools (Union[List[Dict[str, Any]], None]): 可用工具描述。
    label (Union[str, None]): 可选标签。

Returns:
    Tuple: 包含格式化后的 prompt 字典（包括 current_date、db_type、desc、user_query）、history、tools 和 label。
''')

add_english_doc('SqlCall.sql_explain_prompt_hook', '''\
Hook to prepare the prompt for explaining the execution result of a database query.

Args:
    input (Union[str, List, Dict[str, str], None]): A list containing the query and its result.
    history (List[Union[List[str], Dict[str, Any]]]): Conversation history.
    tools (Union[List[Dict[str, Any]], None]): Available tool descriptions.
    label (Union[str, None]): Optional label for the prompt.

Returns:
    Tuple: A tuple containing the formatted prompt dict (history_info, desc, query, result, explain_query), history, tools, and label.
''')

add_chinese_doc('SqlCall.sql_explain_prompt_hook', r'''\ 
为解释数据库查询执行结果准备 prompt 的 hook。

Args:
    input (Union[str, List, Dict[str, str], None]): 包含查询和结果的列表。
    history (List[Union[List[str], Dict[str, Any]]]): 会话历史。
    tools (Union[List[Dict[str, Any]], None]): 可用工具描述。
    label (Union[str, None]): 可选标签。

Returns:
    Tuple: 包含格式化后的 prompt 字典（history_info、desc、query、result、explain_query）、history、tools 和 label。
''')

add_english_doc('SqlCall.extract_sql_from_response', '''\
Extract SQL (or MongoDB pipeline) statement from the raw LLM response.

Args:
    str_response (str): Raw text returned by the LLM which may contain code fences.

Returns:
    tuple[bool, str]: A tuple where the first element indicates whether extraction succeeded, and the second is the cleaned or original content. If sql_post_func is provided, it is applied to the extracted content.
''')

add_chinese_doc('SqlCall.extract_sql_from_response', r'''\ 
从原始 LLM 响应中提取 SQL（或 MongoDB pipeline）语句。

Args:
    str_response (str): LLM 返回的原始文本，可能包含代码块。

Returns:
    tuple[bool, str]: 第一个元素表示是否成功提取，第二个是清洗后的或原始内容。如果提供了 sql_post_func，则会应用于提取结果。
''')

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

如果传入的 'command_or_url' 是一个 URL 字符串（以 'http' 或 'https' 开头），则将连接到远程服务器；否则，将启动并连接到本地服务器。


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
Retrieve the list of tools from the currently connected MCP client.

**Returns:**\n
- Any: The list of tools returned by the MCP client.
''')

add_chinese_doc('MCPClient.list_tools', '''\
获取当前连接的 MCP 客户端的工具列表。

**Returns:**\n
- Any: MCP 客户端返回的工具列表。
''')


add_english_doc('MCPClient.get_tools', '''\
Retrieve a filtered list of tools from the MCP client.

Args:
    allowed_tools (Optional[list[str]]): List of tool names to filter. If None, all tools are returned.

**Returns:**\n
- Any: List of tools that match the filter criteria.
''')

add_chinese_doc('MCPClient.get_tools', '''\
从 MCP 客户端获取经过筛选的工具列表。

Args:
    allowed_tools (Optional[list[str]]): 要筛选的工具名称列表，若为 None，则返回所有工具。

**Returns:**\n
- Any: 符合筛选条件的工具列表。
''')


add_english_doc('MCPClient.deploy', '''\
Deploys the MCP client with the specified SSE server settings asynchronously.

Args:
    sse_settings (SseServerSettings): Configuration settings for the SSE server.
''')

add_chinese_doc('MCPClient.deploy', '''\
使用指定的 SSE 服务器设置异步部署 MCP 客户端。

Args:
    sse_settings (SseServerSettings): SSE 服务器的配置设置。
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


add_english_doc('rag.doc_node.ImageDocNode', '''\
A specialized document node for handling image content in RAG systems.

ImageDocNode extends DocNode to provide specialized functionality for image processing and embedding generation. It automatically handles image loading, base64 encoding for embedding, and PIL Image objects for LLM processing.

Args:
    image_path (str): The file path to the image file. This should be a valid path to an image file (e.g., .jpg, .png, .jpeg).
    uid (Optional[str]): Unique identifier for the document node. If not provided, a UUID will be automatically generated.
    group (Optional[str]): The group name this node belongs to. Used for organizing and filtering nodes.
    embedding (Optional[Dict[str, List[float]]]): Pre-computed embeddings for the image. Keys are embedding model names, values are embedding vectors.
    parent (Optional[DocNode]): Parent node in the document hierarchy. Used for building document trees.
    metadata (Optional[Dict[str, Any]]): Additional metadata associated with the image node.
    global_metadata (Optional[Dict[str, Any]]): Global metadata that applies to all nodes in the document.
    text (Optional[str]): Optional text description or caption for the image.
''')

add_chinese_doc('rag.doc_node.ImageDocNode', '''\
专门用于处理RAG系统中图像内容的文档节点。

ImageDocNode继承自DocNode，为图像处理和嵌入生成提供专门的功能。它自动处理图像加载、用于嵌入的base64编码，以及用于LLM处理的PIL图像对象。

Args:
    image_path (str): 图像文件的文件路径。这应该是一个有效的图像文件路径（例如.jpg、.png、.jpeg）。
    uid (Optional[str]): 文档节点的唯一标识符。如果未提供，将自动生成UUID。
    group (Optional[str]): 此节点所属的组名。用于组织和过滤节点。
    embedding (Optional[Dict[str, List[float]]]): 图像的预计算嵌入。键是嵌入模型名称，值是嵌入向量。
    parent (Optional[DocNode]): 文档层次结构中的父节点。用于构建文档树。
    metadata (Optional[Dict[str, Any]]): 与图像节点关联的附加元数据。
    global_metadata (Optional[Dict[str, Any]]): 适用于文档中所有节点的全局元数据。
    text (Optional[str]): 图像的可选文本描述或标题。
''')

add_example('rag.doc_node.ImageDocNode', '''\
>>> from lazyllm.tools.rag.doc_node import ImageDocNode, MetadataMode
>>> import numpy as np
>>> image_node = ImageDocNode(
...     image_path="/home/mnt/yehongfei/Code/Test/framework.jpg",
...     text="这是一张照片"
)
>>> def clip_emb(content, modality="image"):
...     if modality == "image":
...         return [np.random.rand(512).tolist()]
...     return [np.random.rand(256).tolist()]
>>> embed_functions = {"clip": clip_emb}
>>> image_node.do_embedding(embed_functions)
>>> print(f"嵌入维度: {len(image_node.embedding['clip'])}")
>>> text_representation = image_node.get_text()
>>> content_representation = image_node.get_content(MetadataMode.EMBED)
>>> print(f"text属性: {text_representation}")
>>> print(f"content属性: {content_representation}")    
''')

add_english_doc('rag.doc_node.ImageDocNode.do_embedding', '''\
Generate embeddings for the image using the provided embedding functions.

This method overrides the parent class method to handle image-specific embedding generation. It automatically converts the image to the appropriate format (base64 for embedding) and calls the embedding functions with the image modality.

Args:
    embed (Dict[str, Callable]): Dictionary of embedding functions. Keys are embedding model names, values are callable functions that accept (content, modality) and return embedding vectors.
''')

add_chinese_doc('rag.doc_node.ImageDocNode.do_embedding', '''\
使用提供的嵌入函数为图像生成嵌入。

此方法重写父类方法以处理图像特定的嵌入生成。它自动将图像转换为适当的格式（用于嵌入的base64），并使用图像模态调用嵌入函数。

Args:
    embed (Dict[str, Callable]): 嵌入函数字典。键是嵌入模型名称，值是接受(content, modality)并返回嵌入向量的可调用函数。
''')

add_english_doc('rag.doc_node.ImageDocNode.get_content', '''\
Get the image content in different formats based on the metadata mode.

This method returns the image content in different formats depending on the intended use case. For LLM processing, it returns a PIL Image object. For embedding generation, it returns a base64-encoded image string.

Args:
    metadata_mode (MetadataMode, optional): The mode for content retrieval. Defaults to MetadataMode.LLM.
        - MetadataMode.LLM: Returns PIL Image object for LLM processing
        - MetadataMode.EMBED: Returns base64-encoded image for embedding generation
        - Other modes: Returns the image path as text

**Returns:**\n
- Union[PIL.Image.Image, List[str], str]: The image content in the requested format.
''')

add_chinese_doc('rag.doc_node.ImageDocNode.get_content', '''\
根据元数据模式获取不同格式的图像内容。

此方法根据预期用例返回不同格式的图像内容。对于LLM处理，它返回PIL图像对象。对于嵌入生成，它返回base64编码的图像字符串。

Args:
    metadata_mode (MetadataMode, optional): 内容检索模式。默认为MetadataMode.LLM。
        - MetadataMode.LLM: 返回用于LLM处理的PIL图像对象
        - MetadataMode.EMBED: 返回用于嵌入生成的base64编码图像
        - 其他模式: 返回图像路径作为文本

**Returns:**\n
- Union[PIL.Image.Image, List[str], str]: 请求格式的图像内容。
''')

add_english_doc('rag.doc_node.ImageDocNode.get_text', '''\
Get the image path as text representation.

This method overrides the parent class method to return the image path instead of the content field, since ImageDocNode doesn't use the content field for storing text.

**Returns:**\n
- str: The image file path.
''')

add_chinese_doc('rag.doc_node.ImageDocNode.get_text', '''\
获取图像路径作为文本表示。

此方法重写父类方法以返回图像路径而不是内容字段，因为ImageDocNode不使用内容字段存储文本。

**Returns:**\n
- str: 图像文件路径。
''')

add_english_doc('rag.transform.AdaptiveTransform', '''\
A flexible document transformation system that applies different transforms based on document patterns.

AdaptiveTransform allows you to define multiple transformation strategies and automatically selects the appropriate one based on the document's file path or custom pattern matching. This is particularly useful when you have different types of documents that require different processing approaches.

Args:
    transforms (Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]]): A list of transform configurations or a single transform configuration. 
    num_workers (int, optional): Number of worker threads for parallel processing. Defaults to 0.
''')

add_chinese_doc('rag.transform.AdaptiveTransform', '''\
一个灵活的文档转换系统，根据文档模式应用不同的转换策略。

AdaptiveTransform允许您定义多种转换策略，并根据文档的文件路径或自定义模式匹配自动选择适当的转换方法。当您有不同类型的文档需要不同处理方法时，这特别有用。

Args:
    transforms (Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]]): 转换配置列表或单个转换配置。
    num_workers (int, optional): 并行处理的工作线程数。默认为0。
''')

add_example('rag.transform.AdaptiveTransform', '''\
>>> from lazyllm.tools.rag.transform import AdaptiveTransform, DocNode, SentenceSplitter
>>> doc1 = DocNode(text="这是第一个文档的内容。它包含多个句子。")
>>> doc2 = DocNode(text="这是第二个文档的内容。")
>>> transforms = [
...     {
...         'f': SentenceSplitter,
...         'pattern': '*.txt',
...         'kwargs': {'chunk_size': 50, 'chunk_overlap': 10}
...     },
...     {
...         'f': SentenceSplitter,
...         'pattern': '*.pdf',
...         'kwargs': {'chunk_size': 100, 'chunk_overlap': 20}
...     }
... ]
>>> adaptive = AdaptiveTransform(transforms)
>>> results1 = adaptive.transform(doc1)
>>> print(f"文档1转换结果: {len(results1)} 个块")
>>> for i, result in enumerate(results1):
...     print(f"  块 {i+1}: {result.text}")
>>> results2 = adaptive.transform(doc2)
>>> print(f"文档2转换结果: {len(results2)} 个块")
>>> for i, result in enumerate(results2):
...     print(f"  块 {i+1}: {result.text}")      
''')

add_english_doc('rag.transform.AdaptiveTransform.transform', '''\
Transform a document using the appropriate transformation strategy based on pattern matching.

This method evaluates each transform configuration in order and applies the first one that matches the document's path pattern. The matching logic supports both glob patterns and custom callable functions.

Args:
    document (DocNode): The document node to be transformed.
    **kwargs: Additional keyword arguments passed to the transform function.

**Returns:**\n
- List[Union[str, DocNode]]: A list of transformed results (strings or DocNode objects).
''')

add_chinese_doc('rag.transform.AdaptiveTransform.transform', '''\
根据模式匹配使用适当的转换策略转换文档。

此方法按顺序评估每个转换配置，并应用第一个匹配文档路径模式的转换。匹配逻辑支持glob模式和自定义可调用函数。

Args:
    document (DocNode): 要转换的文档节点。
    **kwargs: 传递给转换函数的附加关键字参数。

**Returns:**\n
- List[Union[str, DocNode]]: 转换结果列表（字符串或DocNode对象）。
''')

add_english_doc('rag.rerank.ModuleReranker', '''\
A reranker that uses trainable modules to reorder documents based on relevance to a query.

ModuleReranker is a specialized reranker that leverages trainable models (such as BGE-reranker, Cohere rerank, etc.) to improve the relevance of retrieved documents. It takes a list of documents and a query, then returns the documents reordered by their relevance scores.

Args:
    name (str): The name of the reranker. Defaults to "ModuleReranker".
    model (Union[Callable, str]): The reranking model. Can be either a model name (string) or a callable function.
    target (Optional[str]): Defaults to None.
    output_format (Optional[str]): The format for output processing. Defaults to None.
    join (Union[bool, str]): Whether to join the results. Defaults to False.
    **kwargs: Additional keyword arguments passed to the reranker model.
''')

add_chinese_doc('rag.rerank.ModuleReranker', '''\
使用可训练模块根据查询相关性重新排序文档的重排序器。

ModuleReranker是一个专门的重排序器，利用可训练模型（如BGE-reranker、Cohere rerank等）来提高检索文档的相关性。它接收文档列表和查询，然后返回按相关性分数重新排序的文档。

Args:
    name (str): 重排序器的名称。默认为"ModuleReranker"。
    model (Union[Callable, str]): 重排序模型。可以是模型名称（字符串）或可调用函数。
    target (Optional[str]): 默认为None。
    output_format (Optional[str]): 输出处理格式。默认为None。
    join (Union[bool, str]): 是否连接结果。默认为False。
    **kwargs: 传递给重排序模型模型的附加关键字参数。
''')

add_example('rag.rerank.ModuleReranker', '''\
>>> from lazyllm.tools.rag.rerank import ModuleReranker, DocNode
>>> def simple_reranker(query, documents, top_n):
...     query_lower = query.lower()
...     scores = []
...     for i, doc in enumerate(documents):
...         score = sum(1 for word in query_lower.split() if word in doc)
...         scores.append((i, score))
...     scores.sort(key=lambda x: x[1], reverse=True)
...     return scores[:top_n]
>>> reranker = ModuleReranker(
...     model=simple_reranker,
...     topk=2
... )
>>> docs = [
...     DocNode(text="机器学习算法在数据分析中应用广泛"),
...     DocNode(text="深度学习模型需要大量训练数据"),
...     DocNode(text="自然语言处理技术发展迅速"),
...     DocNode(text="计算机视觉在自动驾驶中的应用")
... ]
>>> query = "机器学习"
>>> results = reranker.forward(docs, query)
>>> for i, doc in enumerate(results):
...     print(f"  {i+1}. : {doc.text}")
...     print(f"     相关性分数: {doc.relevance_score:.4f}")        
''')

add_english_doc('rag.rerank.ModuleReranker.forward', '''\
Forward pass of the reranker that reorders documents based on relevance to the query.

This method takes a list of documents and a query, then uses the underlying reranking model to score and reorder the documents by relevance. The documents are processed in MetadataMode.EMBED format to ensure compatibility with the reranking model.

Args:
    nodes (List[DocNode]): List of document nodes to be reranked.
    query (str): The query string to rank documents against. Defaults to "".

**Returns:**\n
- List[DocNode]: List of document nodes reordered by relevance score, with relevance_score attribute added.
''')

add_chinese_doc('rag.rerank.ModuleReranker.forward', '''\
重排序器的前向传播，根据与查询的相关性重新排序文档。

此方法接收文档列表和查询，然后使用底层重排序模型对文档进行评分和重新排序。文档以MetadataMode.EMBED格式处理，以确保与重排序模型的兼容性。

Args:
    nodes (List[DocNode]): 要重排序的文档节点列表。
    query (str): 用于排序文档的查询字符串。默认为""。

**Returns:**\n
- List[DocNode]: 按相关性分数重新排序的文档节点列表，添加了relevance_score属性。
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

# agent/functionCall.py
add_agent_chinese_doc('functionCall.StreamResponse', '''\
StreamResponse类用于封装带有前缀和颜色配置的流式输出行为。  
当启用流式模式时，调用实例会将带颜色的文本推送到文件系统队列中，用于异步处理或显示。

Args:
    prefix (str): 输出内容前的前缀文本，通常用于标识信息来源或类别。
    prefix_color (Optional[str]): 前缀文本的颜色，支持终端颜色代码，默认无颜色。
    color (Optional[str]): 主体内容文本颜色，支持终端颜色代码，默认无颜色。
    stream (bool): 是否启用流式输出模式，启用后会将文本推送至文件系统队列，默认关闭。
''')

add_agent_english_doc('functionCall.StreamResponse', '''\
StreamResponse class encapsulates streaming output behavior with configurable prefix and colors.  
When streaming is enabled, calling the instance enqueues colored text to a filesystem queue for asynchronous processing or display.

Args:
    prefix (str): Prefix text before the output, typically used to indicate the source or category.
    prefix_color (Optional[str]): Color of the prefix text, supports terminal color codes, defaults to None.
    color (Optional[str]): Color of the main content text, supports terminal color codes, defaults to None.
    stream (bool): Whether to enable streaming output mode, which enqueues text to the filesystem queue, defaults to False.
''')

add_agent_example('functionCall.StreamResponse', '''\
>>> from lazyllm.tools.agent.functionCall import StreamResponse
>>> resp = StreamResponse(prefix="[INFO]", prefix_color="green", color="white", stream=True)
>>> resp("Hello, world!")
Hello, world!
''')
 
add_chinese_doc('rag.web.DocWebModule', """\
文档Web界面模块，继承自ModuleBase，提供基于Web的文档管理交互界面。

Args:
    doc_server (ServerModule): 文档服务模块实例，提供后端API支持
    title (str): 界面标题，默认为"文档管理演示终端"
    port (int/range/list): 服务端口号或端口范围，默认为20800-20999
    history (list): 初始聊天历史记录，默认为空列表
    text_mode (Mode): 文本处理模式，默认为Mode.Dynamic(动态模式)
    trace_mode (Mode): 追踪模式，默认为Mode.Refresh(刷新模式)

类属性:
    Mode: 模式枚举类，包含:
        - Dynamic: 动态模式
        - Refresh: 刷新模式
        - Appendix: 附录模式

注意事项:
    - 需要配合有效的doc_server实例使用
    - 端口冲突时会自动尝试范围内其他端口
    - 服务停止后会释放相关资源
""")

add_english_doc('rag.web.DocWebModule', """\
Document Web Interface Module, inherits from ModuleBase, provides web-based document management interface.

Args:
    doc_server (ServerModule): Document server module instance providing backend API support
    title (str): Interface title, defaults to "文档管理演示终端"
    port (int/range/list): Service port number or range, defaults to 20800-20999
    history (list): Initial chat history, defaults to empty list
    text_mode (Mode): Text processing mode, defaults to Mode.Dynamic
    trace_mode (Mode): Trace mode, defaults to Mode.Refresh

Class Attributes:
    Mode: Mode enumeration class containing:
        - Dynamic: Dynamic mode
        - Refresh: Refresh mode
        - Appendix: Appendix mode

Notes:
    - Requires a valid doc_server instance to work with
    - Automatically tries other ports in range when port conflict occurs
    - Releases resources when service is stopped
""")

add_chinese_doc('rag.web.DocWebModule.Mode', """\
文档Web模块运行模式枚举类。

取值:
    Dynamic (0): 动态模式，实时更新内容
    Refresh (1): 刷新模式，定期刷新内容
    Appendix (2): 附录模式，将新内容作为附录添加

""")

add_english_doc('rag.web.DocWebModule.Mode', """\
Operation mode enumeration class for DocWebModule.

Values:
    Dynamic (0): Dynamic mode, updates content in real-time
    Refresh (1): Refresh mode, periodically refreshes content
    Appendix (2): Appendix mode, adds new content as appendix

""")


add_example('rag.web.DocWebModule', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.web import DocWebModule
>>> from lazyllm import
>>> doc_server = ServerModule(url="your_url")
>>> doc_web = DocWebModule(
>>>   doc_server=doc_server,
>>>   title="文档管理演示终端",
>>>   port=range(20800, 20805)  # 自动寻找可用端口)
>>> deploy_task = doc_web._get_deploy_tasks()
>>> deploy_task()  
>>> print(doc_web.url)
>>> doc_web.stop()
''')

add_english_doc('rag.web.DocWebModule.wait', '''\
Blocks the current thread to keep the web interface running until manually stopped.

''')

add_chinese_doc('rag.web.DocWebModule.wait', '''\
阻塞当前线程以保持Web界面运行，直到手动停止。

''')

add_english_doc('rag.web.DocWebModule.stop', '''\
Stops the web interface service and releases related resources.

''')

add_chinese_doc('rag.web.DocWebModule.stop', '''\
停止Web界面服务并释放相关资源。

''')

# FuncNodeTransform
add_english_doc('rag.transform.FuncNodeTransform', '''
A wrapper class for user-defined functions that transforms document nodes.

This wrapper supports two modes of operation:
1. When trans_node is False (default): transforms text strings
2. When trans_node is True: transforms DocNode objects

The wrapper can handle various function signatures:
- str -> List[str]: transform=lambda t: t.split('\\\\n')
- str -> str: transform=lambda t: t[:3]
- DocNode -> List[DocNode]: pipeline(lambda x:x, SentenceSplitter)
- DocNode -> DocNode: pipeline(LLMParser)

Args:
    func (Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]]): The user-defined function to be wrapped.
    trans_node (bool, optional): Determines whether the function operates on DocNode objects (True) or text strings (False). Defaults to None.
    num_workers (int): Controls the number of threads or processes used for parallel processing. Defaults to 0.
''')

add_chinese_doc('rag.transform.FuncNodeTransform', '''
用于包装用户自定义函数的转换器类。

此包装器支持两种操作模式：
1. 当 trans_node 为 False（默认）：转换文本字符串
2. 当 trans_node 为 True：转换 DocNode 对象

包装器可以处理各种函数签名：
- str -> List[str]: transform=lambda t: t.split('\\\\n')
- str -> str: transform=lambda t: t[:3]
- DocNode -> List[DocNode]: pipeline(lambda x:x, SentenceSplitter)
- DocNode -> DocNode: pipeline(LLMParser)

Args:
    func (Union[Callable[[str], List[str]], Callable[[DocNode], List[DocNode]]]): 要包装的用户自定义函数。
    trans_node (bool, optional): 确定函数是操作 DocNode 对象（True）还是文本字符串（False）。默认为 None。
    num_workers (int): 控制并行处理的线程/进程数量。默认为 0。
''')

add_example('rag.transform.FuncNodeTransform', '''
>>> import lazyllm
>>> from lazyllm.tools.rag import FuncNodeTransform
>>> from lazyllm.tools import Document, SentenceSplitter

# Example 1: Text-based transformation (trans_node=False)
>>> def split_by_comma(text):
...     return text.split(',')
>>> text_transform = FuncNodeTransform(split_by_comma, trans_node=False)

# Example 2: Node-based transformation (trans_node=True)
>>> def custom_node_transform(node):
...     # Process the DocNode and return a list of DocNodes
...     return [node]  # Simple pass-through
>>> node_transform = FuncNodeTransform(custom_node_transform, trans_node=True)

# Example 3: Using with Document
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="custom", transform=text_transform)
''')

# FuncNodeTransform.transform
add_english_doc('rag.transform.FuncNodeTransform.transform', '''
Transform a document node using the wrapped user-defined function.

This method applies the user-defined function to either the text content of the node (when trans_node=False) or the node itself (when trans_node=True).

Args:
    node (DocNode): The document node to be transformed.
    **kwargs: Additional keyword arguments passed to the transformation function.

Returns:
    List[Union[str, DocNode]]: The transformed results, which can be either strings or DocNode objects depending on the function implementation.
''')

add_chinese_doc('rag.transform.FuncNodeTransform.transform', '''
使用包装的用户自定义函数转换文档节点。

此方法将用户自定义函数应用于节点的文本内容（当 trans_node=False 时）或节点本身（当 trans_node=True 时）。

Args:
    node (DocNode): 要转换的文档节点。
    **kwargs: 传递给转换函数的额外关键字参数。

Returns:
    List[Union[str, DocNode]]: 转换结果，根据函数实现可以是字符串或 DocNode 对象。
''')


add_chinese_doc('rag.web.WebUi', """\
基于 Gradio 的知识库文件管理 Web UI 工具类。

该类用于构建一个简单的 Web 界面，支持创建分组、上传文件、列出/删除分组或文件，并通过 RESTful API 与后端交互。支持快速集成与展示文件管理能力。

Args:
    base_url (str): 后端 API 服务的基础地址。
""")

add_english_doc('rag.web.WebUi', """\
A Gradio-based web UI for managing knowledge base files.

This class provides an interactive UI to create/delete groups, upload files, list files, and perform deletion operations via RESTful APIs. It is designed for rapid integration of file and group management.

Args:
    base_url (str): Base URL of the backend API service.
""")

add_chinese_doc("rag.web.WebUi.basic_headers", '''
生成通用的 HTTP 请求头。

Args:
    content_type (bool): 是否包含 Content-Type 头信息（默认为 True）。

Returns:
    dict: HTTP 请求头字典。
''')

add_english_doc("rag.web.WebUi.basic_headers", '''
Generate standard HTTP headers.

Args:
    content_type (bool): Whether to include Content-Type in the headers (default: True).

Returns:
    dict: Dictionary of HTTP headers.
''')

add_chinese_doc("rag.web.WebUi.muti_headers", '''
生成用于上传文件的 HTTP 请求头。

Returns:
    dict: HTTP 请求头字典。
''')

add_english_doc("rag.web.WebUi.muti_headers", '''
Generate HTTP headers for file upload.

Returns:
    dict: Dictionary of HTTP headers.
''')

add_chinese_doc("rag.web.WebUi.post_request", '''
发送 POST 请求。

Args:
    url (str): 请求地址。
    data (dict): 请求数据，将被转为 JSON。

Returns:
    dict: 响应结果的 JSON。
''')

add_english_doc("rag.web.WebUi.post_request", '''
Send a POST request.

Args:
    url (str): Target request URL.
    data (dict): Request data (will be serialized as JSON).

Returns:
    dict: JSON response from the server.
''')

add_chinese_doc("rag.web.WebUi.get_request", '''
发送 GET 请求。

Args:
    url (str): 请求地址。

Returns:
    dict: 响应结果的 JSON。
''')

add_english_doc("rag.web.WebUi.get_request", '''
Send a GET request.

Args:
    url (str): Target request URL.

Returns:
    dict: JSON response from the server.
''')

add_chinese_doc("rag.web.WebUi.new_group", '''
创建新的文件分组。

Args:
    group_name (str): 分组名称。

Returns:
    str: 创建结果的提示信息。
''')

add_english_doc("rag.web.WebUi.new_group", '''
Create a new file group.

Args:
    group_name (str): Name of the new group.

Returns:
    str: Server message about the creation result.
''')

add_chinese_doc("rag.web.WebUi.delete_group", '''
删除指定的文件分组。

Args:
    group_name (str): 分组名称。

Returns:
    str: 删除结果信息。
''')

add_english_doc("rag.web.WebUi.delete_group", '''
Delete a specific file group.

Args:
    group_name (str): Name of the group to delete.

Returns:
    str: Server message about the deletion.
''')

add_chinese_doc("rag.web.WebUi.list_groups", '''
列出所有文件分组。

Returns:
    List[str]: 分组名称列表。
''')

add_english_doc("rag.web.WebUi.list_groups", '''
List all available file groups.

Returns:
    List[str]: List of group names.
''')

add_chinese_doc("rag.web.WebUi.upload_files", '''
向指定分组上传文件。

Args:
    group_name (str): 分组名称。
    override (bool): 是否覆盖已存在的文件（默认 True）。

Returns:
    Any: 后端返回的上传结果数据。
''')

add_english_doc("rag.web.WebUi.upload_files", '''
Upload files to a specified group.

Args:
    group_name (str): Name of the group.
    override (bool): Whether to override existing files (default: True).

Returns:
    Any: Data returned by the backend.
''')

add_chinese_doc("rag.web.WebUi.list_files_in_group", '''
列出指定分组下的所有文件。

Args:
    group_name (str): 分组名称。

Returns:
    List: 文件信息列表。
''')

add_english_doc("rag.web.WebUi.list_files_in_group", '''
List all files within a specific group.

Args:
    group_name (str): Name of the group.

Returns:
    List: List of file information.
''')

add_chinese_doc("rag.web.WebUi.delete_file", '''
从指定分组中删除文件。

Args:
    group_name (str): 分组名称。
    file_ids (List[str]): 要删除的文件 ID 列表。

Returns:
    str: 删除结果提示。
''')

add_english_doc("rag.web.WebUi.delete_file", '''
Delete specific files from a group.

Args:
    group_name (str): Name of the group.
    file_ids (List[str]): IDs of files to delete.

Returns:
    str: Deletion result message.
''')

add_chinese_doc("rag.web.WebUi.gr_show_list", '''
以 Gradio 表格的形式展示字符串列表。

Args:
    str_list (List): 字符串或子项列表。
    list_name (Union[str, List]): 表头名称或列名列表。

Returns:
    gr.DataFrame: Gradio 表格组件。
''')

add_english_doc("rag.web.WebUi.gr_show_list", '''
Display a list of strings as a Gradio DataFrame.

Args:
    str_list (List): List of strings or rows.
    list_name (Union[str, List]): Column name(s) for the table.

Returns:
    gr.DataFrame: Gradio DataFrame component.
''')

add_chinese_doc("rag.web.WebUi.create_ui", '''
构建基于 Gradio 的文件管理图形界面，包含分组列表、上传、查看、删除等功能标签页。

Returns:
    gr.Blocks: 完整的 Gradio UI 应用实例。
''')

add_english_doc("rag.web.WebUi.create_ui", '''
Build a Gradio-based file management UI, including tabs for group listing, file uploading, viewing, and deletion.

Returns:
    gr.Blocks: A complete Gradio application instance.
''')

add_chinese_doc('rag.index_base.IndexBase.update', '''\
更新索引内容。

该方法接收一组文档节点对象，并将其添加或更新到索引结构中。通常用于增量构建或刷新索引。

Args:
    nodes (List[DocNode]): 需要更新的文档节点列表。
''')

add_english_doc('rag.index_base.IndexBase.update', '''\
Update index contents.

This method receives a list of document nodes and updates or inserts them into the index structure. Typically used for incremental indexing or refreshing data.

Args:
    nodes (List[DocNode]): A list of document nodes to update or insert.
''')

add_chinese_doc('rag.index_base.IndexBase.remove', '''\
从索引中移除指定文档节点。

可根据唯一标识符列表删除索引中的文档节点，可选地指定组名称以限定范围。

Args:
    uids (List[str]): 需要移除的文档节点的唯一标识符列表。
    group_name (Optional[str]): 可选的组名称，用于限定要删除的范围。
''')

add_english_doc('rag.index_base.IndexBase.remove', '''\
Remove specific document nodes from the index.

Removes document nodes based on their unique identifiers, optionally scoped by group name.

Args:
    uids (List[str]): List of unique IDs corresponding to the document nodes to remove.
    group_name (Optional[str]): Optional group name to scope the removal operation.
''')

add_chinese_doc('rag.index_base.IndexBase.query', '''\
执行索引查询。

根据传入的参数执行查询操作，返回匹配的文档节点列表。具体查询逻辑由实现类定义。

Returns:
    List[DocNode]: 查询结果的文档节点列表。
''')

add_english_doc('rag.index_base.IndexBase.query', '''\
Execute a query over the index.

Performs a query based on the given arguments and returns matching document nodes. The logic depends on the specific implementation.

Returns:
    List[DocNode]: A list of matched document nodes from the index.
''')
