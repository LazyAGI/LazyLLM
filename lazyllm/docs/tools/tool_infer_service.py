# flake8: noqa E501
import importlib
import functools
from .. import utils
add_infer_service_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.infer_service'))
add_infer_service_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.infer_service'))
add_infer_service_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools.infer_service'))

add_infer_service_chinese_doc('InferServer', """\
推理服务服务器类，继承自ServerBase。

提供模型推理服务的创建、管理、监控和日志查询等RESTful API接口。

""")

add_infer_service_english_doc('InferServer', """\
Inference service server class, inherits from ServerBase.

Provides RESTful API interfaces for model inference service creation, management, monitoring and log query.

""")


add_infer_service_chinese_doc('InferServer.create_job', """\
创建推理任务。

根据任务描述创建新的模型推理服务，启动部署线程并初始化任务状态。

Args:
    job (JobDescription): 任务描述对象
    token (str): 用户令牌

Returns:
    dict: 包含任务ID的响应
""")

add_infer_service_english_doc('InferServer.create_job', """\
Create inference task.

Create new model inference service based on job description, start deployment thread and initialize task status.

Args:
    job (JobDescription): Job description object
    token (str): User token

Returns:
    dict: Response containing job ID
""")

add_infer_service_chinese_doc('InferServer.cancel_job', """\
取消推理任务。

停止指定的推理任务，清理资源并更新任务状态。

Args:
    job_id (str): 任务ID
    token (str): 用户令牌

Returns:
    dict: 包含任务状态的响应
""")

add_infer_service_english_doc('InferServer.cancel_job', """\
Cancel inference task.

Stop specified inference task, clean up resources and update task status.

Args:
    job_id (str): Job ID
    token (str): User token

Returns:
    dict: Response containing task status
""")

add_infer_service_chinese_doc('InferServer.list_jobs', """\
列出所有推理任务。

获取当前用户的所有推理任务列表。

Args:
    token (str): 用户令牌

Returns:
    dict: 任务列表信息
""")

add_infer_service_english_doc('InferServer.list_jobs', """\
List all inference tasks.

Get all inference tasks list for current user.

Args:
    token (str): User token

Returns:
    dict: Task list information
""")

add_infer_service_chinese_doc('InferServer.get_job_info', """\
获取任务详细信息。

查询指定任务的详细信息，包括状态、端点、耗时等。

Args:
    job_id (str): 任务ID
    token (str): 用户令牌

Returns:
    dict: 任务详细信息
""")

add_infer_service_english_doc('InferServer.get_job_info', """\
Get task detailed information.

Query detailed information of specified task, including status, endpoint, cost time, etc.

Args:
    job_id (str): Job ID
    token (str): User token

Returns:
    dict: Task detailed information
""")

add_infer_service_chinese_doc('InferServer.get_job_log', """\
获取任务日志。

获取指定任务的日志文件路径或日志内容。

Args:
    job_id (str): 任务ID
    token (str): 用户令牌

Returns:
    dict: 日志信息
""")

add_infer_service_english_doc('InferServer.get_job_log', """\
Get task log.

Get log file path or log content of specified task.

Args:
    job_id (str): Job ID
    token (str): User token

Returns:
    dict: Log information
""")

