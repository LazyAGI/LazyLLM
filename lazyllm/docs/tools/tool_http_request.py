# flake8: noqa E501
import importlib
import functools
from .. import utils
add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools'))
add_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools'))

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse', """\
HTTP执行器响应类，用于封装和处理HTTP请求的响应结果。

提供对HTTP响应内容的统一访问接口，支持文件类型检测和内容提取。

Args:
    response (httpx.Response, optional): httpx库的响应对象，默认为None


**Returns:**\n
- HttpExecutorResponse实例，提供多种响应内容访问方式
""")

add_english_doc('http_request.http_executor_response.HttpExecutorResponse', """\
HTTP executor response class for encapsulating and processing HTTP request response results.

Provides unified access interface for HTTP response content, supporting file type detection and content extraction.

Args:
    response (httpx.Response, optional): httpx library response object, defaults to None

**Returns:**\n
- HttpExecutorResponse instance, providing multiple response content access methods
""")

add_chinese_doc('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
获取HTTP响应的内容类型。

从响应头中提取 'content-type' 字段的值，用于判断响应内容的类型。

**Returns:**\n
- str: 响应的内容类型，如果未找到则返回空字符串。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.get_content_type', '''\
Get the content type of the HTTP response.

Extracts the 'content-type' field value from the response headers to determine the type of response content.

**Returns:**\n
- str: The content type of the response, or empty string if not found.
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

**Returns:**\n
- tuple[str, bytes]: 包含内容类型和文件二进制数据的元组。如果不是文件类型，则返回空字符串和空字节。
''')

add_english_doc('http_request.http_executor_response.HttpExecutorResponse.extract_file', '''\
Extract file content from HTTP response.

If the response content type is file-related (such as image, audio, video), extracts the content type and binary data of the file.

**Returns:**\n
- tuple[str, bytes]: A tuple containing the content type and binary data of the file. If not a file type, returns empty string and empty bytes.
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

