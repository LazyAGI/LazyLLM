# flake8: noqa E501
import importlib
import functools
from .. import utils
add_tools_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.tools'))
add_tools_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.tools'))
add_tools_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools.tools'))

add_tools_chinese_doc("Weather", """
天气信息查询工具类，继承自HttpTool。

提供城市天气信息的实时查询功能，通过中国气象局API获取指定城市的天气数据。
""")

add_tools_english_doc("Weather", """
Weather information query tool class, inherits from HttpTool.

Provides real-time weather information query functionality, retrieves weather data for specified cities through China Meteorological Administration API.
""")

add_tools_example("Weather", """
from lazyllm.tools.tools import Weather

weather = Weather()
""")

add_tools_chinese_doc("Weather.forward", """
查询某个城市的天气。接收的城市输入最小范围为地级市，如果是直辖市则最小范围为区。输入的城市或区名称不带后缀的“市”或者“区”。参考下面的例子。

Args:
    city_name (str): 需要获取天气的城市名称。

**Returns:**\n
- Optional[Dict]: 天气信息的字典数据，如果城市不存在返回None
""")

add_tools_english_doc("Weather.forward", """
Query the weather of a specific city. The minimum input scope for cities is at the prefecture level, and for municipalities, it is at the district level. The input city or district name should not include the suffix "市" (city) or "区" (district). Refer to the examples below.

Args:
    city_name (str): The name of the city for which weather information is needed.

**Returns:**\n
- Optional[Dict]: Dictionary containing weather information, returns None if city doesn't exist
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
简单计算器模块，继承自ModuleBase。

提供数学表达式计算功能，支持基本的算术运算和数学函数。
''')

add_tools_english_doc('Calculator', '''
Simple calculator module, inherits from ModuleBase.

Provides mathematical expression calculation functionality, supports basic arithmetic operations and math functions.
''')

add_tools_example('Calculator', '''
from lazyllm.tools.tools import Calculator
calc = Calculator()
''')

add_tools_chinese_doc('Calculator.forward', '''
计算用户输入的表达式的值。

Args:
    exp (str): 需要计算的表达式的值。必须符合 Python 计算表达式的语法。可使用 Python math 库中的数学函数。
    *args: 可变位置参数
    **kwargs: 可变关键字参数
''')

add_tools_english_doc('Calculator.forward', '''
Calculate the value of the user input expression.

Args:
    exp (str): The expression to be calculated. It must conform to the syntax for evaluating expressions in Python. Mathematical functions from the Python math library can be used.
    *args: Variable positional arguments
    **kwargs: Variable keyword arguments
''')

add_tools_example('Calculator.forward', '''
from lazyllm.tools.tools import Calculator
calc = Calculator()
result1 = calc.forward("2 + 3 * 4")
print(f"2 + 3 * 4 = {result1}")
''')

add_tools_chinese_doc('TencentSearch', '''
腾讯搜索接口封装类，用于调用腾讯云的内容搜索服务。

提供对腾讯云搜索API的封装，支持关键词搜索和结果处理。

Args:
    secret_id (str): 腾讯云API密钥ID，用于身份认证
    secret_key (str): 腾讯云API密钥，用于身份认证
''')

add_tools_english_doc('TencentSearch', '''
Tencent search interface wrapper class for calling Tencent Cloud content search services.

Provides encapsulation of Tencent Cloud search API, supporting keyword search and result processing.

Args:
    secret_id (str): Tencent Cloud API key ID for authentication
    secret_key (str): Tencent Cloud API key for authentication

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

**Returns:**\n
- package: 包含搜索结果的对象，如果发生错误则返回空package
''')

add_tools_english_doc('TencentSearch.forward', '''
Searches for the query entered by the user.

Args:
    query (str): The content that the user wants to query.

**Returns:**\n
- package: Object containing search results, returns empty package if error occurs
''')

add_tools_example('TencentSearch.forward', '''
from lazyllm.tools.tools import TencentSearch
secret_id = '<your_secret_id>'
secret_key = '<your_secret_key>'
searcher = TencentSearch(secret_id, secret_key)
res = searcher('calculus')
''')

add_tools_chinese_doc('JsonExtractor', '''
JSON提取器，用于从文本中提取JSON数据。

Args:
    base_model (LLMBase): 语言模型
    schema (Union[str, Dict[str, Any]]): JSON结构，可以是JSON字符串或字典。示例：'{"name": "", "age": 0, "city": ""}' 或 {"name": "", "age": 0, "city": ""}
    field_descriptions (Union[str, Dict[str, str]], optional): 字段描述，可以是字符串或字典。如果字典，键是字段名称，值是字段描述。示例：{"name": "姓名", "age": "年龄", "city": "城市"}

Returns:
    Union[Dict[str, Any], List[Dict[str, Any]]]: 提取的JSON数据，如果有多个，则返回列表。如果提取失败则返回空字典。
''')

add_tools_english_doc('JsonExtractor', '''
JSON extractor for extracting JSON data from text.

Args:
    base_model (LLMBase): Language model
    schema (Union[str, Dict[str, Any]]): JSON structure, can be a JSON string or dict. Example: '{"name": "", "age": 0, "city": ""}' or {"name": "", "age": 0, "city": ""}
    field_descriptions (Union[str, Dict[str, str]], optional): Field descriptions, can be a string or dict. If dict, keys are field names and values are descriptions. Example: {"name": "Name", "age": "Age", "city": "City"}

Returns:
    Union[Dict[str, Any], List[Dict[str, Any]]]: Extracted JSON data, returns list if there are multiple, returns empty dictionary if extraction fails.
''')

add_tools_example('JsonExtractor', '''
>>> from lazyllm.tools.tools import JsonExtractor
>>> from lazyllm import OnlineChatModule
>>> llm = lazyllm.OnlineChatModule()
>>> extractor = JsonExtractor(llm, schema='{"name": "", "age": 0, "city": ""}', field_descriptions={'name': '姓名', 'age': '年龄', 'city': '城市'})
>>> res = extractor("张三的年龄是20岁，住在北京; 李四的年龄是25岁，住在上海")
>>> print(res)
[{'name': '张三', 'age': 20, 'city': '北京'}, {'name': '李四', 'age': 25, 'city': '上海'}]
''')

add_tools_chinese_doc('JsonConcentrator', '''
JSON聚合器，用于将多个JSON数据聚合成一个JSON数据。

Args:
    base_model (LLMBase): 语言模型
    schema (Union[str, Dict[str, Any]]): JSON结构，可以是JSON字符串或字典。示例：'{"name": "", "age": 0, "city": ""}' 或 {"name": "", "age": 0, "city": ""}
''')

add_tools_english_doc('JsonConcentrator', '''
JSON concentrator for aggregating multiple JSON data into a single JSON data.

Args:
    base_model (LLMBase): Language model
    schema (Union[str, Dict[str, Any]]): JSON structure, can be a JSON string or dict. Example: '{"name": "", "age": 0, "city": ""}' or {"name": "", "age": 0, "city": ""}
''')

add_tools_example('JsonConcentrator', '''
>>> from lazyllm.tools.tools import JsonConcentrator
>>> from lazyllm import OnlineChatModule
>>> llm = lazyllm.OnlineChatModule()
>>> concentrator = JsonConcentrator(llm, schema='{"name": "", "age": 0, "city": ""}')
>>> res = concentrator([{'name': '张三', 'age': 20, 'city': '北京'}, {'name': '李四', 'age': 25, 'city': '上海'}])
>>> print(res)
{'name': '张三,李四', 'age': 20 - 25, 'city': '北京,上海'}
''')
# ---------------------------------------------------------------------------- #

# mcp/client.py

