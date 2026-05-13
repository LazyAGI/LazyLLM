# flake8: noqa E501
import importlib
import functools
from .. import utils
add_services_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.services'))
add_services_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.services'))
add_services_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools.services'))

add_services_chinese_doc('client.ClientBase', '''\
客户端基类，用于管理服务连接和状态转换。

Args:
    url (str): 服务端点的URL地址。

属性：
    url: 服务端点的URL地址。
''')

add_services_english_doc('client.ClientBase', '''\
Base client class for managing service connections and status conversions.

Args:
    url (str): URL of the service endpoint.

Attributes:
    url: URL of the service endpoint.
''')

add_services_chinese_doc('client.ClientBase.uniform_status', '''\
统一化任务状态字符串。

Args:
    status (str): 原始状态字符串。

**Returns:**\n
- str: 标准化的状态字符串，可能的值包括：
    - 'Invalid': 无效状态
    - 'Ready': 就绪状态
    - 'Done': 完成状态
    - 'Cancelled': 已取消状态
    - 'Failed': 失败状态
    - 'Running': 运行中状态
    - 'Pending': 等待中状态（包括TBSubmitted、InQueue、Pending）
''')

add_services_english_doc('client.ClientBase.uniform_status', '''\
Standardize task status string.

Args:
    status (str): Original status string.

**Returns:**\n
- str: Standardized status string, possible values include:
    - 'Invalid': Invalid status
    - 'Ready': Ready status
    - 'Done': Completed status
    - 'Cancelled': Cancelled status
    - 'Failed': Failed status
    - 'Running': Running status
    - 'Pending': Pending status (includes TBSubmitted, InQueue, Pending)
''')

