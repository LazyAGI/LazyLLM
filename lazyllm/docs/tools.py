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
>>> tools = ["get_current_weather", "get_n_day_weather_forecast"]
>>> tm = ToolManager(tools)
>>> tm([{'name': 'get_n_day_weather_forecast', 'arguments': {'location': 'Beijing', 'num_days': 3}}])
'{"location": "Beijing", "temperature": "85", "unit": "fahrenheit", "num_days": 3}'
""")

add_chinese_doc('FunctionCall', '''\
FunctionCall是单轮工具调用类，如果LLM中的信息不足以回答用户的问题，必需结合外部知识来回答用户问题，则调用该类。如果LLM输出需要工具调用，则进行工具调用，并输出工具调用结果，输出结果为List类型，包含当前轮的输入、模型输出、工具输出。如果不需要工具调用，则直接输出LLM结果，输出结果为string类型。

Args:
    llm (ModuleBase): 要使用的LLM可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM使用的工具名称列表。
''')

add_english_doc('FunctionCall', '''\
FunctionCall is a single-round tool call class. If the information in LLM is not enough to answer the uesr's question, it is necessary to combine external knowledge to answer the user's question. If the LLM output required a tool call, the tool call is performed and the tool call result is output. The output result is of List type, including the input, model output, and tool output of the current round. If a tool call is not required, the LLM result is directly output, and the output result is of string type.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[str]): A list of tool names for LLM to use.
''')

add_example('FunctionCall', """\
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
>>> llm = lazyllm.Trainable("internlm2-chat-20b").start()   # or llm = lazyllm.OnlineChatModule(source="sensenova")
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

add_example('ReWOOAgent', """\
>>> @fc_register("tool")
>>> def WikipediaWorker(input: str):
...     '''
...     Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.
... 
...     Args:
...         input (str): search query.
...     '''
...     docstore = DocstoreExplorer(Wikipedia())
...     tool = Tool(name="Search", func=docstore.search, description="useful for when you need to ask with search")
...     LOG.info(f"wikipedia input: {input}")
...     evidence = tool.run(input)
...     LOG.info(f"wikipedia output: {evidence}")
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
...     llm = lazyllm.OnlineChatModule(source="openai")
...     query = f"Respond in short directly with no extra words.\n\n{input}"
...     LOG.info(f"llm query: {query}, input: {input}")
...     response = llm(query, llm_chat_history=[])
...     LOG.info(f"llm res: {response}")
...     return response
...
>>> tools = ["WikipediaWorker", "LLMWorker"]
>>> llm = lazyllm.TrainableModule("GLM-4-9B-Chat").deploy_method(deploy.vllm).start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = ReWOOAgent(llm, tools)
>>> query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
>>> res = agent(query)
>>> print(res)
'\nHennessy '
""")
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
