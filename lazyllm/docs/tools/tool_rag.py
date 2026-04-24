# flake8: noqa E501
import importlib
import functools
from .. import utils
add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools'))
add_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools'))


def _add_bilingual_docs(doc_map):
    for obj_name, (zh_doc, en_doc) in doc_map.items():
        add_chinese_doc(obj_name, zh_doc)
        add_english_doc(obj_name, en_doc)

add_english_doc('Document', '''\
Initialize a document management module with optional embedding, storage, and user interface.

The ``Document`` module provides a unified interface for managing document datasets, including support for local files, cloud-based files, or temporary document files. It can optionally run with a document manager service or a web UI, and supports multiple embedding models and custom storage backends.

Args:
    dataset_path (Optional[str]): Path to the dataset directory. If not found, the system will attempt to locate it in ``lazyllm.config["data_path"]``.
    embed (Optional[Union[Callable, Dict[str, Callable]]]): Embedding function or mapping of embedding functions. When a dictionary is provided, keys are embedding names and values are embedding models.
    create_ui (bool, optional): Whether to create the document-management UI. It requires an available ``DocServer`` and can be combined with ``manager=True`` or ``manager=DocServer(...)``.
    manager (Union[bool, str, DocServer, DocumentProcessor], optional): Document manager mode. ``True`` launches a local ``DocServer`` together with a local parsing service. ``DocServer(...)`` connects an existing document-management service. ``DocumentProcessor(...)`` connects a parsing service only and requires a non-map ``store_conf``. ``'ui'`` is accepted as a compatibility alias for ``manager=True, create_ui=True``.
    server (Union[bool, int], optional): Whether to run a server interface for knowledge bases. ``True`` enables a default server, an integer specifies a custom port, and ``False`` disables it. Defaults to ``False``.
    name (Optional[str]): Name identifier for this document collection. Defaults to the system default name.
    launcher (Optional[Launcher]): Launcher instance for managing server processes. Defaults to a remote asynchronous launcher.
    doc_files (Optional[List[str]]): Temporary document files. When used, ``dataset_path`` must be ``None``. Only MapStore is supported in this mode.
    doc_fields (Optional[Dict[str, DocField]]): Metadata field configuration for storing and retrieving document attributes.
    store_conf (Optional[Dict]): Storage configuration. Defaults to in-memory MapStore.
    display_name (Optional[str]): Human-readable display name for this document module. Defaults to the collection name.
    description (Optional[str]): Description of the document collection. Defaults to ``"algorithm description"``.
    schema_extractor (Optional[Union[LLMBase, SchemaExtractor]]): Optional schema extractor used for metadata schema analysis and registration.
    enable_path_monitoring (Optional[bool]): Whether to watch the local dataset path for file additions and removals. Defaults to enabled only for local documents without ``DocServer``/``DocumentProcessor`` manager mode.
''')

add_chinese_doc('Document', '''\
初始化一个文档管理模块，支持可选的向量化、存储和用户界面。

``Document`` 模块提供了统一的文档数据集管理接口，支持本地文件、云端文件或临时文档文件。它可以选择运行文档管理服务或 Web UI，并支持多种向量化模型和自定义存储后端。

Args:
    dataset_path (Optional[str]): 数据集目录路径。如果路径不存在，系统会尝试在 ``lazyllm.config["data_path"]`` 中查找。
    embed (Optional[Union[Callable, Dict[str, Callable]]]): 文档向量化函数或函数字典。若为字典，键为 embedding 名称，值为对应的模型。
    create_ui (bool, optional): 是否创建文档管理 UI。该能力要求当前存在可用的 ``DocServer``，可与 ``manager=True`` 或 ``manager=DocServer(...)`` 组合使用。
    manager (Union[bool, str, DocServer, DocumentProcessor], optional): 文档管理模式。``True`` 表示启动本地 ``DocServer`` 及其 parsing service；``DocServer(...)`` 表示连接已有文档管理服务；``DocumentProcessor(...)`` 表示仅连接解析服务，此时必须提供非 map 的 ``store_conf``；``'ui'`` 仅作为 ``manager=True, create_ui=True`` 的兼容写法保留。
    server (Union[bool, int], optional): 是否为知识库运行服务接口。``True`` 表示启动默认服务；整型数值表示自定义端口；``False`` 表示关闭。默认为 ``False``。
    name (Optional[str]): 文档集合的名称标识符。默认为系统默认名称。
    launcher (Optional[Launcher]): 启动器实例，用于管理服务进程。默认使用远程异步启动器。
    doc_files (Optional[List[str]]): 临时文档文件列表。当使用此参数时，``dataset_path`` 必须为 ``None``，且仅支持 MapStore。
    doc_fields (Optional[Dict[str, DocField]]): 元数据字段配置，用于存储和检索文档属性。
    store_conf (Optional[Dict]): 存储配置。默认使用内存中的 MapStore。
    display_name (Optional[str]): 文档模块的可读显示名称。默认为集合名称。
    description (Optional[str]): 文档集合的描述。默认为 ``"algorithm description"``。
    schema_extractor (Optional[Union[LLMBase, SchemaExtractor]]): 可选 schema extractor，用于元数据 schema 分析与注册。
    enable_path_monitoring (Optional[bool]): 是否监控本地数据目录的文件新增和删除。仅在未接入 ``DocServer`` / ``DocumentProcessor`` 的本地模式下默认开启。
''')

add_english_doc('DocServer', '''\
Primary entry point of the document service.

``DocServer`` manages document upload/add/reparse/delete flows, task tracking, knowledge-base management,
chunk inspection, and cross-kb transfer. It is the recommended replacement for the legacy ``DocManager`` /
``DocListManager`` APIs.

Args:
    port (Optional[int]): Local service port when starting an in-process server.
    url (Optional[str]): Existing doc_service URL. When provided, the instance works as a remote client.
    parser_url (Optional[str]): Parsing service URL used by the local doc_service instance.
    db_config (Optional[Dict[str, Any]]): Metadata database configuration for doc_service.
    parser_db_config (Optional[Dict[str, Any]]): Parsing task database configuration for the parsing service.
    parser_poll_interval (float): Poll interval used by local parser coordination.
    storage_dir (Optional[str]): Local storage directory for uploaded files.
    callback_url (Optional[str]): Callback URL used to receive parsing task updates.
    launcher: Launcher used to start local services.
''')

add_chinese_doc('DocServer', '''\
文档服务的主入口。

``DocServer`` 负责文档上传/添加/重解析/删除、任务跟踪、知识库管理、chunk 查看，以及跨知识库文档转移。
它是 legacy ``DocManager`` / ``DocListManager`` API 的推荐替代方案。

Args:
    port (Optional[int]): 本地启动服务时使用的端口。
    url (Optional[str]): 已存在的 doc_service 地址；提供后当前实例作为远程客户端使用。
    parser_url (Optional[str]): 本地 doc_service 使用的 parsing service 地址。
    db_config (Optional[Dict[str, Any]]): doc_service 元数据数据库配置。
    parser_db_config (Optional[Dict[str, Any]]): parsing service 任务数据库配置。
    parser_poll_interval (float): 本地解析协调使用的轮询间隔。
    storage_dir (Optional[str]): 上传文件保存目录。
    callback_url (Optional[str]): 接收解析任务回调的地址。
    launcher: 本地服务启动器。
''')

add_english_doc('DocServer.add', '''\
Add existing local files through the ``/v1/docs/add`` endpoint.

Use this method when the file paths are already accessible on the DocServer host. The request body is an
``AddRequest`` containing ``kb_id``, ``algo_id``, and ``items``. Each item can provide ``file_path``,
optional ``doc_id``, and optional ``metadata``.

**Returns:**
    Standard API response. ``data["items"]`` contains the accepted ``doc_id`` and asynchronous ``task_id``.
''')

add_chinese_doc('DocServer.add', '''\
通过 ``/v1/docs/add`` 接口添加服务端可直接访问的本地文件。

当文件路径已经对 DocServer 所在机器可见时，使用该方法。请求体为 ``AddRequest``，包含 ``kb_id``、``algo_id``
和 ``items``。每个 item 可提供 ``file_path``，以及可选的 ``doc_id``、``metadata``。

**Returns:**
    标准 API 响应。``data["items"]`` 中包含接受后的 ``doc_id`` 和异步 ``task_id``。
''')

add_english_doc('DocServer.upload', '''\
Upload files into DocServer-managed storage through the ``/v1/docs/upload`` flow.

Use this method when you want DocServer to manage uploaded copies of the source files. The request body is an
``UploadRequest`` with ``kb_id``, ``algo_id``, and ``items``. Each item uses ``file_path`` as the local source
path and can optionally include ``doc_id`` or ``metadata``.

**Returns:**
    Standard API response. ``data["items"]`` contains the accepted ``doc_id`` and asynchronous ``task_id``.
''')

add_chinese_doc('DocServer.upload', '''\
通过 ``/v1/docs/upload`` 流程将文件上传到 DocServer 管理的存储目录。

当你希望由 DocServer 保存上传副本时，使用该方法。请求体为 ``UploadRequest``，包含 ``kb_id``、``algo_id``
和 ``items``。每个 item 使用 ``file_path`` 作为本地源路径，也可以附带可选的 ``doc_id``、``metadata``。

**Returns:**
    标准 API 响应。``data["items"]`` 中包含接受后的 ``doc_id`` 和异步 ``task_id``。
''')

add_english_doc('DocServer.reparse', '''\
Reparse existing documents through the ``/v1/docs/reparse`` endpoint.

The request body is a ``ReparseRequest`` with ``kb_id`` and ``doc_ids``. Use it after metadata
or parsing configuration changes when you want to enqueue new parse tasks for existing documents.

Specify either ``algo_id`` to reparse all node groups of that algorithm, or ``reparse_group``
(a node-group name) to reparse a single group. The two fields are mutually exclusive — providing
both raises a validation error. When neither is provided the first algorithm bound to the
knowledge base is used and all its node groups are reparsed.
''')

add_chinese_doc('DocServer.reparse', '''\
通过 ``/v1/docs/reparse`` 接口重新解析已有文档。

请求体为 ``ReparseRequest``，包含 ``kb_id`` 和 ``doc_ids``。当元数据或解析配置变更后，
需要为已有文档重新入队解析任务时，可使用该方法。

可通过 ``algo_id`` 指定重解析该算法下的所有节点组，或通过 ``reparse_group``（节点组名称）
指定仅重解析某一个节点组。两个字段互斥，同时传入会触发校验错误。若两者均不传，则使用知识库绑定的
第一个算法，并重解析其所有节点组。
''')

add_english_doc('ReparseRequest', '''\
Request model for the ``/v1/docs/reparse`` endpoint.

Args:
    doc_ids (List[str]): IDs of the documents to reparse. Must not be empty.
    kb_id (str): Knowledge-base ID. Defaults to ``"__default__"``.
    algo_id (Optional[str]): Algorithm ID whose node groups should all be reparsed.
        Mutually exclusive with ``reparse_group``. When both are ``None`` the first
        algorithm bound to the knowledge base is used.
    reparse_group (Optional[str]): Name of a single node group to reparse.
        Mutually exclusive with ``algo_id``. The owning algorithm is resolved
        automatically from the knowledge-base bindings.
    idempotency_key (Optional[str]): Optional idempotency key for deduplication.

Raises:
    ValueError: If ``doc_ids`` is empty, or if both ``algo_id`` and ``reparse_group``
        are provided at the same time.
''')

add_chinese_doc('ReparseRequest', '''\
``/v1/docs/reparse`` 接口的请求模型。

Args:
    doc_ids (List[str]): 需要重解析的文档 ID 列表，不能为空。
    kb_id (str): 知识库 ID，默认为 ``"__default__"``。
    algo_id (Optional[str]): 指定算法 ID，将重解析该算法下的所有节点组。
        与 ``reparse_group`` 互斥。两者均为 ``None`` 时，使用知识库绑定的第一个算法。
    reparse_group (Optional[str]): 指定单个节点组的名称，仅重解析该节点组。
        与 ``algo_id`` 互斥，所属算法会从知识库绑定关系中自动推断。
    idempotency_key (Optional[str]): 可选的幂等键，用于请求去重。

Raises:
    ValueError: ``doc_ids`` 为空，或同时传入了 ``algo_id`` 和 ``reparse_group`` 时抛出。
''')

add_english_doc('DocServer.delete', '''\
Delete documents from a knowledge base through the ``/v1/docs/delete`` endpoint.

The request body is a ``DeleteRequest`` with ``kb_id`` and ``doc_ids``. Deletion is asynchronous,
so the returned ``task_id`` should be tracked through the task APIs when you need final status.

All algorithms bound to the knowledge base are handled automatically — there is no need to
specify an ``algo_id``. If any algorithm's parse task is in WORKING state the request is
rejected with ``E_STATE_CONFLICT``. WAITING add-tasks are cancelled before the delete proceeds.
''')

add_chinese_doc('DocServer.delete', '''\
通过 ``/v1/docs/delete`` 接口从知识库中删除文档。

请求体为 ``DeleteRequest``，包含 ``kb_id`` 和 ``doc_ids``。删除是异步操作，因此如果需要最终状态，
应继续通过任务接口跟踪返回的 ``task_id``。

知识库下绑定的所有算法均会被自动处理，无需指定 ``algo_id``。若任意算法的解析任务处于 WORKING
状态，请求会被拒绝并返回 ``E_STATE_CONFLICT``；处于 WAITING 状态的添加任务会在删除前被自动取消。
''')

add_english_doc('DeleteRequest', '''\
Request model for the ``/v1/docs/delete`` endpoint.

Args:
    doc_ids (List[str]): IDs of the documents to delete. Must not be empty.
    kb_id (str): Knowledge-base ID. Defaults to ``"__default__"``.
    idempotency_key (Optional[str]): Optional idempotency key for deduplication.

Raises:
    ValueError: If ``doc_ids`` is empty.
''')

add_chinese_doc('DeleteRequest', '''\
``/v1/docs/delete`` 接口的请求模型。

Args:
    doc_ids (List[str]): 需要删除的文档 ID 列表，不能为空。
    kb_id (str): 知识库 ID，默认为 ``"__default__"``。
    idempotency_key (Optional[str]): 可选的幂等键，用于请求去重。

Raises:
    ValueError: ``doc_ids`` 为空时抛出。
''')

add_english_doc('DocServer.patch_metadata', '''\
Patch document metadata through the ``/v1/docs/metadata/patch`` endpoint.

The request body is a ``MetadataPatchRequest`` with ``kb_id``, ``algo_id``, and ``items``. Each item targets one
document and carries a partial metadata patch in ``patch``.
''')

add_chinese_doc('DocServer.patch_metadata', '''\
通过 ``/v1/docs/metadata/patch`` 接口更新文档元数据。

请求体为 ``MetadataPatchRequest``，包含 ``kb_id``、``algo_id`` 和 ``items``。每个 item 指向一个文档，
并在 ``patch`` 中携带需要合并的局部元数据。
''')

add_english_doc('DocServer.get_task', '''\
Get one task record through the ``/v1/tasks/{task_id}`` endpoint.

Args:
    task_id (str): Task ID returned by add, upload, reparse, delete, transfer, or metadata patch operations.

**Returns:**
    Standard API response with the current task status and task payload.
''')

add_chinese_doc('DocServer.get_task', '''\
通过 ``/v1/tasks/{task_id}`` 接口获取单个任务记录。

Args:
    task_id (str): add、upload、reparse、delete、transfer 或 metadata patch 等操作返回的任务 ID。

**Returns:**
    包含当前任务状态和任务负载的标准 API 响应。
''')

add_english_doc('DocServer.cancel_task', '''\
Cancel a waiting task through the ``/v1/tasks/cancel`` endpoint.

Args:
    task_id (str): Task ID to cancel.

**Returns:**
    Standard API response indicating whether the task was canceled successfully.
''')

add_chinese_doc('DocServer.cancel_task', '''\
通过 ``/v1/tasks/cancel`` 接口取消一个处于等待中的任务。

Args:
    task_id (str): 要取消的任务 ID。

**Returns:**
    表示任务是否取消成功的标准 API 响应。
''')

add_english_doc('DocServer.list_chunks', '''\
List parsed chunks for a document through the ``/v1/chunks`` endpoint.

Args:
    kb_id (str): Knowledge-base ID.
    doc_id (str): Source document ID.
    group (str): Node group name to inspect.
    algo_id (str): Algorithm ID.
    page (int): 1-based page number.
    page_size (int): Number of chunks per page.
    offset (Optional[int]): Explicit offset. When omitted, the service derives it from ``page`` and ``page_size``.

Returns:
    Paginated chunk data including ``items`` and ``total``.
''')

add_chinese_doc('DocServer.list_chunks', '''\
通过 ``/v1/chunks`` 接口分页查看文档的解析 chunk。

Args:
    kb_id (str): 知识库 ID。
    doc_id (str): 文档 ID。
    group (str): 要查看的节点组名。
    algo_id (str): 算法 ID。
    page (int): 从 1 开始的页码。
    page_size (int): 每页 chunk 数量。
    offset (Optional[int]): 显式偏移量；未传时服务端会根据 ``page`` 和 ``page_size`` 推导。

Returns:
    包含 ``items`` 与 ``total`` 的分页结果。
''')

add_english_doc('DocServer.transfer', '''\
Transfer parsed documents between knowledge bases under the same algorithm.

The request body is a ``TransferRequest``. Each transfer item must provide a unique ``target_doc_id`` in the target
knowledge base. Transfer across different algorithms is not supported. Optional ``target_filename`` and
``target_file_path`` can override the destination file name/path recorded for the transferred document.
''')

add_chinese_doc('DocServer.transfer', '''\
在同一算法下的不同知识库之间转移已解析文档。

请求体为 ``TransferRequest``。每个转移项都必须在目标知识库中提供唯一的 ``target_doc_id``。
当前不支持跨算法 transfer。可选字段 ``target_filename`` 与 ``target_file_path`` 用于覆盖目标文档记录的文件名或文件路径。
''')

_add_bilingual_docs({
    'rag.doc_service.base.DocServiceError': (
        'Doc service 使用的结构化业务异常。',
        'Structured business exception used by doc_service.'
    ),
    'rag.doc_service.base.AddFileItem.validate_file_path': (
        '校验单个文件项中的 ``file_path`` 字段。',
        'Validate the ``file_path`` field of one file item.'
    ),
    'rag.doc_service.base.DocItemsRequest.validate_items': (
        '校验文档 item 列表不能为空。',
        'Validate that the document item list is not empty.'
    ),
    'rag.doc_service.base.TransferRequest.validate_items': (
        '校验 transfer 请求中的 item 列表及其基本约束。',
        'Validate transfer request items and their basic constraints.'
    ),
    'rag.doc_service.base.MetadataPatchRequest.validate_items': (
        '校验 metadata patch 请求中的 item 列表。',
        'Validate metadata patch request items.'
    ),
    'rag.doc_service.base.KbDeleteBatchRequest.validate_kb_ids': (
        '校验批量删除知识库请求中的 ``kb_ids``。',
        'Validate ``kb_ids`` in a batch knowledge-base delete request.'
    ),
    'rag.doc_service.base.TaskBatchRequest.validate_task_ids': (
        '校验批量任务查询请求中的 ``task_ids``。',
        'Validate ``task_ids`` in a batch task query request.'
    ),
    'rag.doc_service.base.KbBatchQueryRequest.validate_kb_ids': (
        '校验批量知识库查询请求中的 ``kb_ids``。',
        'Validate ``kb_ids`` in a batch knowledge-base query request.'
    ),
    'rag.parsing_service.base.AddDocRequest.normalize_deprecated_fields': (
        '将旧版 add-doc 字段归一化到当前请求格式。',
        'Normalize deprecated add-doc fields into the current request format.'
    ),
    'rag.parsing_service.base.UpdateMetaRequest.normalize_deprecated_fields': (
        '将旧版 update-meta 字段归一化到当前请求格式。',
        'Normalize deprecated update-meta fields into the current request format.'
    ),
    'rag.parsing_service.base.DeleteDocRequest.normalize_legacy_fields': (
        '将旧版 delete-doc 字段归一化到当前请求格式。',
        'Normalize legacy delete-doc fields into the current request format.'
    ),
    'rag.parsing_service.server.DocumentProcessor.set_callback_url': (
        '更新 parsing service 发送任务回调时使用的 callback URL。',
        'Update the callback URL used by the parsing service for task callbacks.'
    ),
    'rag.transform.base.NodeTransform.transform': (
        '将单个文档节点转换为零个或多个输出节点。',
        'Transform one document node into zero or more output nodes.'
    ),
    'rag.doc_service.doc_manager.DocManager': (
        'Doc service 的核心业务实现，负责文档、任务、知识库以及 parser 协调。',
        'Core doc_service implementation for document, task, knowledge-base, and parser coordination.'
    ),
    'rag.doc_service.parser_client.ParserClient': (
        '用于访问 parsing service HTTP 接口的轻量客户端。',
        'Lightweight client for parsing-service HTTP endpoints.'
    ),
})

_add_bilingual_docs({
    f'rag.doc_service.doc_manager.DocManager.{name}': docs
    for name, docs in {
        'set_callback_url': (
            '设置运行时任务回调 URL。',
            'Set the runtime task callback URL.'
        ),
        'run_idempotent': (
            '在幂等键保护下执行一次 doc service 变更操作。',
            'Execute one doc_service mutation under idempotency-key protection.'
        ),
        'upload': (
            '接收上传文件并为其创建异步解析任务。',
            'Accept uploaded files and create asynchronous parse tasks for them.'
        ),
        'add_files': (
            '登记服务端可见文件并为其创建异步解析任务。',
            'Register server-visible files and create asynchronous parse tasks for them.'
        ),
        'reparse': (
            '为已有文档重新入队解析任务。',
            'Enqueue new parse tasks for existing documents.'
        ),
        'delete': (
            '从知识库中删除文档并创建异步删除任务。',
            'Delete documents from a knowledge base and create asynchronous delete tasks.'
        ),
        'transfer': (
            '在同一算法下跨知识库转移文档。',
            'Transfer documents between knowledge bases under the same algorithm.'
        ),
        'patch_metadata': (
            '对已有文档执行局部 metadata 更新。',
            'Apply partial metadata updates to existing documents.'
        ),
        'on_task_callback': (
            '处理 parser service 回调的任务状态更新。',
            'Handle task-state callbacks sent by the parsing service.'
        ),
        'list_docs': (
            '分页列出知识库中的文档。',
            'List documents in a knowledge base with pagination.'
        ),
        'get_doc_detail': (
            '查询单个文档的详细信息。',
            'Return detailed information for one document.'
        ),
        'list_tasks': (
            '按状态分页列出任务记录。',
            'List task records by status with pagination.'
        ),
        'get_task': (
            '按任务 ID 获取单个任务记录。',
            'Get one task record by task ID.'
        ),
        'get_tasks_batch': (
            '批量获取多个任务记录。',
            'Fetch multiple task records in one batch.'
        ),
        'cancel_task': (
            '取消一个尚未执行完成的任务。',
            'Cancel a task that has not finished executing.'
        ),
        'list_algorithms': (
            '列出 parsing service 提供的算法。',
            'List algorithms exposed by the parsing service.'
        ),
        'get_algo_groups': (
            '获取指定算法的节点组信息。',
            'Get node-group information for one algorithm.'
        ),
        'list_algorithms_compat': (
            '以兼容旧客户端的格式返回算法列表。',
            'Return the algorithm list in a legacy-compatible format.'
        ),
        'get_algorithm_info': (
            '获取指定算法的详细信息。',
            'Get detailed information for one algorithm.'
        ),
        'list_chunks': (
            '分页查看指定文档的解析 chunk。',
            'List parsed chunks for one document with pagination.'
        ),
        'health': (
            '执行 doc service 健康检查。',
            'Run a health check for doc_service.'
        ),
        'list_kbs': (
            '分页列出知识库。',
            'List knowledge bases with pagination.'
        ),
        'get_kb': (
            '获取单个知识库的信息。',
            'Get information for one knowledge base.'
        ),
        'batch_get_kbs': (
            '批量获取多个知识库的信息。',
            'Fetch information for multiple knowledge bases in one batch.'
        ),
        'create_kb': (
            '创建新的知识库。',
            'Create a new knowledge base.'
        ),
        'update_kb': (
            '更新已有知识库的元信息。',
            'Update metadata of an existing knowledge base.'
        ),
        'unbind_algo': (
            '从知识库解绑一个算法，并异步清理该算法独有节点组的解析数据；与其他算法共享的节点组数据保留。',
            'Unbind one algorithm from a knowledge base and asynchronously clean up parse data '
            'for node groups exclusive to that algorithm; node groups shared with other algorithms are preserved.'
        ),
        'delete_kb': (
            '删除一个知识库。',
            'Delete one knowledge base.'
        ),
        'delete_kbs': (
            '批量删除多个知识库。',
            'Delete multiple knowledge bases in one batch.'
        ),
    }.items()
})

_add_bilingual_docs({
    f'rag.doc_service.parser_client.ParserClient.{name}': docs
    for name, docs in {
        'health': (
            '检查 parsing service 是否可用。',
            'Check whether the parsing service is healthy.'
        ),
        'add_doc': (
            '向 parsing service 提交新增文档任务。',
            'Submit an add-document task to the parsing service.'
        ),
        'update_meta': (
            '向 parsing service 提交 metadata 更新任务。',
            'Submit a metadata-update task to the parsing service.'
        ),
        'delete_doc': (
            '向 parsing service 提交删除文档任务。',
            'Submit a delete-document task to the parsing service.'
        ),
        'cancel_task': (
            '请求 parsing service 取消一个任务。',
            'Ask the parsing service to cancel a task.'
        ),
        'list_algorithms': (
            '获取 parsing service 当前暴露的算法列表。',
            'Get the list of algorithms currently exposed by the parsing service.'
        ),
        'get_algorithm_groups': (
            '获取指定算法的节点组信息。',
            'Get node-group information for one algorithm.'
        ),
        'list_doc_chunks': (
            '获取指定文档在某个节点组下的 chunk 列表。',
            'Get parsed chunks for one document under a specific node group.'
        ),
    }.items()
})

_add_bilingual_docs({
    f'rag.doc_service.doc_server.DocServer.{name}': docs
    for name, docs in {
        'build_openapi_app': (
            '构建用于导出 OpenAPI 的 FastAPI 应用对象。',
            'Build the FastAPI application used for OpenAPI export.'
        ),
        'build_openapi_schema': (
            '生成 doc service 的 OpenAPI schema。',
            'Generate the OpenAPI schema for doc_service.'
        ),
        'export_openapi': (
            '将 doc service 的 OpenAPI schema 导出到文件。',
            'Export the doc_service OpenAPI schema to a file.'
        ),
        'list_docs': (
            '分页列出知识库中的文档。',
            'List documents in a knowledge base with pagination.'
        ),
        'get_doc': (
            '获取单个文档的详细信息。',
            'Get detailed information for one document.'
        ),
        'list_tasks': (
            '分页列出任务记录。',
            'List task records with pagination.'
        ),
        'get_tasks_batch': (
            '批量获取多个任务记录。',
            'Fetch multiple task records in one batch.'
        ),
        'get_task_info': (
            '获取单个任务记录。',
            'Get one task record.'
        ),
        'set_runtime_callback_url': (
            '更新运行时任务回调 URL。',
            'Update the runtime task callback URL.'
        ),
        'list_kbs': (
            '分页列出知识库。',
            'List knowledge bases with pagination.'
        ),
        'get_kb': (
            '获取单个知识库的信息。',
            'Get information for one knowledge base.'
        ),
        'list_algorithms': (
            '列出可用算法。',
            'List available algorithms.'
        ),
        'get_algorithm_info': (
            '获取指定算法的详细信息。',
            'Get detailed information for one algorithm.'
        ),
        'create_kb': (
            '创建新的知识库。',
            'Create a new knowledge base.'
        ),
        'update_kb': (
            '更新知识库的元信息。',
            'Update knowledge-base metadata.'
        ),
        'batch_get_kbs': (
            '批量获取多个知识库的信息。',
            'Fetch information for multiple knowledge bases in one batch.'
        ),
        'unbind_algo': (
            '从知识库解绑一个算法，并异步清理该算法独有节点组的解析数据；与其他算法共享的节点组数据保留。',
            'Unbind one algorithm from a knowledge base and asynchronously clean up parse data '
            'for node groups exclusive to that algorithm; node groups shared with other algorithms are preserved.'
        ),
        'delete_kb': (
            '删除一个知识库。',
            'Delete one knowledge base.'
        ),
        'delete_kbs': (
            '批量删除多个知识库。',
            'Delete multiple knowledge bases in one batch.'
        ),
    }.items()
})

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

add_chinese_doc('Document.list_all_files_in_directory', """\
列出指定目录路径中的所有文件。

该方法会以递归或非递归方式遍历目录并收集所有文件路径。可以选择跳过隐藏文件和目录（以 “.” 开头的）。如果传入的路径本身是文件，则返回仅包含该文件路径的列表。

Args:
    dataset_path (str): 要列出文件列表的目录。
    skip_hidden_path (bool, optional): 是否跳过隐藏文件和目录（以 “.” 开头）。默认值为 True
    recursive (bool, optional): 是否递归搜索子目录。如果为 False，则只返回当前目录下的文件。默认值为 True。

**Returns:**\n
- List[str]: 绝对文件路径列表。如果路径不存在或不是目录，则返回空列表。
""")

add_english_doc('Document.list_all_files_in_directory', """\
List all files in a given directory path.

This method recursively or non-recursively traverses a directory and collects all file paths. It can optionally skip hidden files and directories (those starting with '.'). If the provided path is a file instead of a directory, it returns a list containing only that file path.

Args:
    dataset_path (str): The path to the directory to list.
    skip_hidden_path (bool, optional): Whether to skip hidden files and directories (those starting with '.'). Defaults to True.
    recursive (bool, optional): Whether to recursively search subdirectories. If False, only files in the immediate directory are returned. Defaults to True.

**Returns:**\n
- List[str]: A list of absolute file paths. Returns an empty list if the path does not exist or is not a directory.
""")

add_chinese_doc('Document.connect_sql_manager', """\
连接 SQL 管理器并初始化文档与数据库的映射处理器。

此方法会验证数据库连接，并根据传入的文档表模式（schema）更新或重置数据库表结构。如果已存在的 schema 与新传入的 schema 不一致，则需要设置 ``force_refresh=True`` 以强制刷新。

Args:
    sql_manager (SqlManager): SQL 管理器实例，用于连接和操作数据库。
    schma (Optional[DocInfoSchema]): 文档表模式定义。包含字段名称、类型及描述。
    force_refresh (bool, optional): 当 schema 发生变化时，是否强制刷新数据库表结构。默认为 ``True``。

Raises:
    RuntimeError: 当数据库连接失败时抛出。
    AssertionError: 当未提供 schema 或 schema 变更时未设置 ``force_refresh`` 抛出。
""")

add_english_doc('Document.connect_sql_manager', """\
Connect to the SQL manager and initialize the document-to-database processor.

This method validates the database connection and updates or resets the database table schema based on the provided document schema. If the existing schema differs from the new one, ``force_refresh=True`` must be set to enforce a reset.

Args:
    sql_manager (SqlManager): SQL manager instance for database connection and operations.
    schma (Optional[DocInfoSchema]): Document table schema definition, including field names, types, and descriptions.
    force_refresh (bool, optional): Whether to force refresh the database schema when changes are detected. Defaults to ``True``.

Raises:
    RuntimeError: If the database connection fails.
    AssertionError: If schema is missing or schema change occurs without setting ``force_refresh``.
""")

add_chinese_doc('Document.get_sql_manager', """\
获取当前文档模块绑定的 SQL 管理器实例。

**Returns:**\n
- SqlManager: 已连接的 SQL 管理器实例。
""")

add_english_doc('Document.get_sql_manager', """\
Get the SQL manager instance currently bound to this document module.

**Returns:**\n
- SqlManager: The connected SQL manager instance.
""")

add_chinese_doc('Document.extract_db_schema', """\
基于文档数据集和大语言模型自动提取数据库表模式（schema）。

此方法会扫描数据集中的所有文件，并调用大语言模型提取文档信息结构。可选择是否打印提取的 schema。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于解析文档并提取 schema 的模型。
    print_schema (bool, optional): 是否在日志中打印提取的 schema。默认为 ``False``。

**Returns:**\n
- DocInfoSchema: 提取的数据库表模式。
""")

add_english_doc('Document.extract_db_schema', """\
Extract the database schema from the dataset using a large language model.

This method scans all files in the dataset and uses the LLM to extract document information schema. Optionally, the schema can be printed to the logs.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): Model used to parse documents and extract schema.
    print_schema (bool, optional): Whether to log the extracted schema. Defaults to ``False``.

**Returns:**\n
- DocInfoSchema: The extracted database schema.
""")

add_chinese_doc('Document.update_database', """\
使用大语言模型解析文档并将提取的信息更新到数据库。

此方法会遍历数据集中的所有文件，提取文档结构化信息，并将其写入数据库。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于解析文档并提取信息的大语言模型。
""")

add_english_doc('Document.update_database', """\
Update the database with information extracted from documents using a large language model.

This method iterates through all files in the dataset, extracts structured information, and exports it into the database.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): Model used to parse documents and extract information.
""")

add_chinese_doc('Document.create_kb_group', """\
创建一个新的知识库分组（KB Group），并返回绑定到该分组的文档对象。

知识库分组用于在同一个文档模块中划分不同的文档集合，每个分组可以有独立的字段定义和存储配置。

Args:
    name (str): 知识库分组的名称。
    doc_fields (Optional[Dict[str, DocField]]): 文档字段定义。指定每个字段的名称、类型和描述。
    store_conf (Optional[Dict]): 存储配置，用于定义存储后端及其参数。

**Returns:**\n
- Document: 一个绑定到新建知识库分组的文档对象副本。
""")

add_english_doc('Document.create_kb_group', """\
Create a new knowledge base group (KB Group) and return a document object bound to that group.

Knowledge base groups are used to partition different document collections within the same document module. Each group can have independent field definitions and storage configurations.

Args:
    name (str): Name of the knowledge base group.
    doc_fields (Optional[Dict[str, DocField]]): Document field definitions, specifying field names, types, and descriptions.
    store_conf (Optional[Dict]): Storage configuration, defining the backend and its parameters.

**Returns:**\n
- Document: A copy of the document object bound to the newly created knowledge base group.
""")

add_chinese_doc('Document.activate_group', """\
激活指定的知识库分组，并可选择指定要启用的 embedding key。

激活后，文档模块会在该分组下执行检索和存储操作。如果未指定 embedding key，则默认启用所有可用的 embedding。

Args:
    group_name (str): 要激活的知识库分组名称。
    embed_keys (Optional[Union[str, List[str]]]): 需要启用的 embedding key，可以是单个字符串或字符串列表。默认为空列表，表示启用全部 embedding。
""")

add_english_doc('Document.activate_group', """\
Activate the specified knowledge base group, optionally enabling specific embedding keys.

After activation, the document module will perform retrieval and storage operations within the given group. If no embedding keys are provided, all available embeddings will be enabled by default.

Args:
    group_name (str): Name of the knowledge base group to activate.
    embed_keys (Optional[Union[str, List[str]]]): Embedding keys to enable, either as a string or a list of strings. Defaults to an empty list, enabling all embeddings.
""")

add_chinese_doc('Document.activate_groups', """\
批量激活多个知识库分组。

该方法会依次调用 `activate_group` 来激活传入的所有分组。

Args:
    groups (Union[str, List[str]]): 要激活的分组名称或分组名称列表。
""")

add_english_doc('Document.activate_groups', """\
Activate multiple knowledge base groups in batch.

This method iteratively calls `activate_group` to activate all the provided groups.

Args:
    groups (Union[str, List[str]]): A single group name or a list of group names to activate.
""")

add_chinese_doc('Document.get_store', """\
获取存储占位符对象。

该方法返回一个存储层的占位符，用于延迟绑定具体的存储实现。调用者可以基于此对象进行存储相关的配置或扩展。

**Returns:**\n
- StorePlaceholder: 存储占位符对象。
""")

add_english_doc('Document.get_store', """\
Get the storage placeholder object.

This method returns a placeholder for the storage layer, allowing deferred binding of the actual storage implementation.
The caller can use this object for storage-related configuration or extension.

**Returns:**\n
- StorePlaceholder: Storage placeholder object.
""")

add_chinese_doc('Document.get_embed', """\
获取 embedding 占位符对象。

该方法返回一个 embedding 层的占位符，用于延迟绑定具体的 embedding 实现。调用者可以基于此对象进行 embedding 相关的配置或扩展。

**Returns:**\n
- EmbedPlaceholder: embedding 占位符对象。
""")

add_english_doc('Document.get_embed', """\
Get the embedding placeholder object.

This method returns a placeholder for the embedding layer, allowing deferred binding of the actual embedding implementation.
The caller can use this object for embedding-related configuration or extension.

**Returns:**\n
- EmbedPlaceholder: Embedding placeholder object.
""")

add_chinese_doc('Document.register_index', """\
注册索引类型。

该方法允许用户为文档模块注册新的索引类型，以便扩展检索能力。注册后，可以通过索引类型来调用对应的索引实现。

Args:
    index_type (str): 索引类型的名称。
    index_cls (IndexBase): 索引类，需继承自 ``IndexBase``。
    *args: 初始化索引类时的可变参数。
    **kwargs: 初始化索引类时的关键字参数。
""")

add_english_doc('Document.register_index', """\
Register a new index type.

This method allows users to register a new index type for the document module, enabling extension of retrieval capabilities.
Once registered, the index can be referenced by its type.

Args:
    index_type (str): Name of the index type.
    index_cls (IndexBase): Index class, must inherit from ``IndexBase``.
    *args: Variable arguments for index initialization.
    **kwargs: Keyword arguments for index initialization.
""")

add_chinese_doc('Document.find', """\
查找目标。

该方法返回一个可调用对象，用于执行目标查找操作。它会延迟调用底层实现以获取指定的目标对象。

Args:
    target: 需要查找的目标。

**Returns:**\n
- Callable: 可调用对象，用于执行目标查找。
""")

add_english_doc('Document.find', """\
Find the target.

This method returns a callable object that performs a deferred lookup operation.
It invokes the underlying implementation to retrieve the specified target.

Args:
    target: The target to be found.

**Returns:**\n
- Callable: Callable object for performing the target lookup.
""")

add_chinese_doc('Document.clear_cache', """\
清理缓存。

该方法用于清理文档模块的缓存，可以指定要清理的分组名称列表。如果未指定分组名称，则默认清理所有分组的缓存。

Args:
    group_names (Optional[List[str]]): 需要清理缓存的分组名称列表。默认为 ``None``，表示清理全部缓存。
""")

add_english_doc('Document.clear_cache', """\
Clear cache.

This method clears the cache of the document module. A list of group names can be specified to
clear cache for specific groups. If no group names are provided, all group caches will be cleared.

Args:
    group_names (Optional[List[str]]): List of group names whose cache should be cleared.
        Defaults to ``None``, meaning clear all caches.
""")

add_english_doc('Document.create_node_group', '''
Generate a node group produced by the specified rule.

Args:
    name (str): The name of the node group.
    transform (Callable): The transformation rule that converts a node into a node group. The function prototype is `(DocNode, group_name, **kwargs) -> List[DocNode]`. Currently built-in options include [SentenceSplitter][lazyllm.tools.SentenceSplitter], and users can define their own transformation rules.
    trans_node (bool): Determines whether the input and output of transform are `DocNode` or `str`, default is None. Can only be set to true when `transform` is `Callable`.
    num_workers (int): number of new threads used for transform. default: 0
    parent (str): The node that needs further transformation. The series of new nodes obtained after transformation will be child nodes of this parent node. If not specified, the transformation starts from the root node.
    ref (str): The name of another node group to reference. The referenced node group must be a descendant of the parent. During transformation, nodes from the referenced node group are passed to the transform function as the `ref` parameter (if the transform function supports it).
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
    ref (str): 当前节点组引用的其他节点组名称。引用的节点组必须是父节点组的后代。在转换时，ref 指定的节点组中的相关节点会作为参数传递给 transform 函数（如果 transform 函数支持 ref 参数）。
    kwargs: 和具体实现相关的参数。
''')

add_example('Document.create_node_group', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> # Example with ref parameter: create a node group that references another group
>>> documents.create_node_group(name="fine_chunks", parent="sentences",
...                             transform=SentenceSplitter, chunk_size=128, chunk_overlap=12)
>>> def transform_with_ref(text, ref):
...     # ref contains nodes from the referenced group
...     return "\n".join(ref)
>>> documents.create_node_group(name="summary_chunks", parent="sentences",
...                             transform=transform_with_ref, ref="fine_chunks")
''')

add_chinese_doc('Document.find_parent', """\
查找目标的父节点。

该方法返回一个可调用对象，用于执行父节点查找操作。它会延迟调用底层实现以获取指定目标的父节点。

Args:
    target: 需要查找父节点的目标。

**Returns:**\n
- Callable: 可调用对象，用于执行父节点查找。
""")

add_english_doc('Document.find_parent', """\
Find the parent node of the target.

This method returns a callable object that performs a deferred parent lookup operation.
It invokes the underlying implementation to retrieve the parent node of the specified target.

Args:
    target: The target for which to find the parent.

**Returns:**\n
- Callable: Callable object for performing the parent lookup.
""")

add_example('Document.find_parent', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
>>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
>>> documents.find_parent('children')
''')

add_chinese_doc('Document.find_children', """\
查找目标的子节点。

该方法返回一个可调用对象，用于执行子节点查找操作。它会延迟调用底层实现以获取指定目标的所有子节点。

Args:
    target: 需要查找子节点的目标。

**Returns:**\n
- Callable: 可调用对象，用于执行子节点查找。
""")

add_english_doc('Document.find_children', """\
Find the children nodes of the target.

This method returns a callable object that performs a deferred children lookup operation.
It invokes the underlying implementation to retrieve all children nodes of the specified target.

Args:
    target: The target for which to find the children.

**Returns:**\n
- Callable: Callable object for performing the children lookup.
""")

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
>>> doc1 = Document(dataset_path="your_files_path")
>>> doc2 = Document(dataset_path="your_files_path")
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
>>> doc1 = Document(dataset_path="your_files_path")
>>> doc2 = Document(dataset_path="your_files_path")
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

add_chinese_doc('Document.drop_algorithm', '''
用于删除当前文档集合的在文档解析服务中注册的算法信息。
''')

add_english_doc('Document.drop_algorithm', '''
Delete the algorithm information registered in the document parsing service for the current document collection.
''')

add_chinese_doc('Document.analyze_schema_by_llm', '''
用于使用大模型为文档管理模块中特定的知识库或文档集合自动抽取字段集合，返回自动生成的Pydantic Model。
支持传入特定知识库id和文档id列表。

Args:
    kb_id: 目标知识库id
    doc_ids: 目标文档id列表
''')

add_english_doc('Document.analyze_schema_by_llm', '''
Use an LLM to auto-infer a field schema for a specific knowledge base or document set in the Document manager, returning a generated Pydantic model. Supports narrowing the sample by kb_id and a list of doc_ids.

Args:
    kb_id: Target knowledge base id.
    doc_ids: List of target document ids.
''')

add_chinese_doc('Document.register_schema_set', '''
手动注册一个 Pydantic Model 作为当前算法的字段集合（schema），并绑定到指定知识库。
如果该知识库已绑定其他 schema，默认会报错；传入 ``force_refresh=True`` 则会替换旧绑定并清理旧数据。

Args:
    schema_set (Type[BaseModel]): 要注册的 Pydantic 模型，用作 schema 定义。
    kb_id (Optional[str]): 目标知识库 ID，默认为 ``DEFAULT_KB_ID``。
    force_refresh (bool): 若已有绑定，是否强制刷新并覆盖。默认 ``False``。

Returns:
    str: 生成的 schema_set_id。
''')

add_english_doc('Document.register_schema_set', '''
Manually register a Pydantic model as the schema for the current algorithm and bind it to a specific knowledge base.
If the KB is already bound to another schema, it raises by default; set ``force_refresh=True`` to replace the binding and clean old records.

Args:
    schema_set (Type[BaseModel]): Pydantic model that defines the schema to register.
    kb_id (Optional[str]): Target knowledge base ID. Defaults to ``DEFAULT_KB_ID``.
    force_refresh (bool): Whether to force refresh when a binding already exists. Defaults to ``False``.

Returns:
    str: The generated ``schema_set_id``.
''')

add_chinese_doc('Document.get_nodes', '''\
按条件获取节点列表。

Args:
    uids (Optional[List[str]]): 指定节点 uid 列表。
    doc_ids (Optional[Set]): 指定文档 id 集合。
    group (Optional[str]): 节点组名。
    kb_id (Optional[str]): 知识库 id。
    numbers (Optional[Set]): 节点编号集合。

**Returns:**\n
- List[DocNode]: 命中的节点列表。
''')

add_english_doc('Document.get_nodes', '''\
Get nodes by criteria.

Args:
    uids (Optional[List[str]]): List of node uids to fetch.
    doc_ids (Optional[Set]): Set of document ids to filter by.
    group (Optional[str]): Node group name.
    kb_id (Optional[str]): Knowledge base id.
    numbers (Optional[Set]): Set of node numbers.

**Returns:**\n
- List[DocNode]: Matched nodes.
''')

add_example('Document.get_nodes', '''\
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> doc = Document()
>>> nodes = doc.get_nodes(doc_ids={'doc_1'}, group='CoarseChunk', kb_id='kb_1', numbers={1, 2})
''')

add_chinese_doc('Document.get_window_nodes', '''\
获取指定节点在同一文档内的窗口节点。

Args:
    node (DocNode): 目标节点。
    span (tuple[int, int]): 窗口范围，基于 node.number 的相对偏移。
    merge (bool): 是否将窗口节点合并为一个节点返回。

**Returns:**\n
- Union[List[DocNode], DocNode]: 窗口节点列表，或合并后的单节点。
''')

add_english_doc('Document.get_window_nodes', '''\
Get window nodes around a target node within the same document.

Args:
    node (DocNode): Target node.
    span (tuple[int, int]): Window range based on relative offsets of node.number.
    merge (bool): Whether to merge window nodes into a single node.

**Returns:**\n
- Union[List[DocNode], DocNode]: Window nodes list or a merged node.
''')

add_example('Document.get_window_nodes', '''\
>>> import lazyllm
>>> from lazyllm.tools import Document
>>> doc = Document()
>>> node = doc.get_nodes(doc_ids={'doc_1'}, group='CoarseChunk', kb_id='kb_1', numbers={10})[0]
>>> window_nodes = doc.get_window_nodes(node, span=(-2, 2), merge=False)
''')



# rag/graph_document.py

add_english_doc('GraphDocument', '''\
GraphRAG-based document processing module for knowledge graph querying.

This class provides a high-level interface for working with GraphRAG (Graph-based Retrieval-Augmented Generation) on top of a Document instance. It manages the GraphRAG service lifecycle, including knowledge graph initialization, indexing, and querying capabilities.

Args:
    document (Document): The Document instance to build the knowledge graph from. The GraphRAG knowledge graph will be created in ``{document._manager._dataset_path}/.graphrag_kg``.
''')

add_chinese_doc('GraphDocument', '''\
基于 GraphRAG 的知识图谱查询文档处理模块。

此类在 Document 实例基础之上提供了用于操作 GraphRAG（基于图的检索增强生成）的高级接口。它负责管理 GraphRAG 服务的生命周期，包括知识图谱的初始化、索引构建以及查询等能力。

Args:
    document (Document): 用于构建知识图谱的 Document 实例。GraphRAG 的知识图谱将被创建在 ``{document._manager._dataset_path}/.graphrag_kg`` 路径下。
''')

add_example('GraphDocument', '''\
>>> import lazyllm
>>> from lazyllm.tools import Document, GraphDocument, GraphRetriever
>>> doc = Document(dataset_path='your_doc_path', name='test_graphrag')
>>> graph_document = GraphDocument(doc)
>>> graph_document.start()
>>> user_input = input('Press Enter when files are ready in dataset path')
>>> graph_document.init_graphrag_kg(regenerate_config=True)
>>> # Now you need to edit $dataset_path/.graphrag_kg/settings.yaml
>>> user_input = input('Press Enter when settings.yaml is ready')
>>> graph_document.start_graphrag_index(override=True)
>>> status_dict = graph_document.graphrag_index_status()
>>> lazyllm.LOG.info(f'graphrag index status: {status_dict}')
>>> # Wait until the index is completed
>>> user_input = input('Press Enter to start graphrag retriever: ')
>>> graph_retriever = GraphRetriever(graph_document)
>>> your_query = input('Enter your query: ')
>>> print(graph_retriever.forward(your_query))
''')

add_english_doc('GraphDocument.init_graphrag_kg', '''
Initialize the GraphRAG knowledge graph directory and prepare files. This method copies all files from the document dataset to the GraphRAG input directory and initializes the GraphRAG project structure. The files are renamed with UUID suffixes to avoid naming conflicts.

Args:
    regenerate_config (bool, optional): Whether to regenerate the GraphRAG configuration files. If True, existing configuration will be overwritten. Defaults to True.
''')

add_chinese_doc('GraphDocument.init_graphrag_kg', '''
初始化 GraphRAG 知识图谱目录并准备相关文件。该方法会将文档数据集中所有文件复制到 GraphRAG 的输入目录，并初始化 GraphRAG 的项目结构。文件会被追加 UUID 后缀以避免命名冲突。

Args:
    regenerate_config (bool, optional): 是否重新生成 GraphRAG 的配置文件。如果为 True，将覆盖已有的配置。默认值为 True。
''')

add_english_doc('GraphDocument.start_graphrag_index', '''
Start the GraphRAG indexing process. This method initiates the asynchronous indexing task that builds the knowledge graph from the prepared files. The indexing runs in the background and can be monitored using graphrag_index_status().

Args:
    override (bool, optional): Whether to override existing index if it exists. If True, any existing index will be deleted and recreated. Defaults to True.
''')

add_chinese_doc('GraphDocument.start_graphrag_index', '''
启动 GraphRAG 的索引构建过程。该方法会启动一个异步索引任务，根据已准备好的文件构建知识图谱。索引过程在后台运行，可通过 graphrag_index_status() 进行监控。

Args:
    override (bool, optional): 如果已有索引，是否覆盖重建。为 True 时会删除并重新创建现有索引。默认值为 True。
''')

add_english_doc('GraphDocument.graphrag_index_status', '''
Get the status of the current GraphRAG indexing task.

**Returns:**\n
- dict: A dictionary containing the indexing task status information.
''')

add_chinese_doc('GraphDocument.graphrag_index_status', '''
获取当前 GraphRAG 索引任务的状态。

**Returns:**\n
- dict: 包含索引任务状态信息的字典。
''')

add_english_doc('GraphDocument.query', '''
Query the GraphRAG knowledge graph. This method performs a query against the indexed knowledge graph and returns an answer based on the graph structure and relationships.

Args:
    query (str): The natural language query to search the knowledge graph.

**Returns:**\n
- str: The answer to the query.
''')

add_chinese_doc('GraphDocument.query', '''
查询 GraphRAG 知识图谱。此方法会对已建立索引的知识图谱执行查询，并基于图的结构和关系返回答案。

Args:
    query (str): 用于搜索知识图谱的自然语言查询。

**Returns:**\n
- str: 查询问题的答案。
''')

add_english_doc('UrlGraphDocument', '''\
A lightweight wrapper for querying remote GraphRAG services via URL.

This class provides a simplified interface to query remote GraphRAG services that are already deployed and running.

Args:
    graphrag_url (str): The base URL of the remote GraphRAG service endpoint. Should be in the format 'http://hostname:port'.
''')

add_chinese_doc('UrlGraphDocument', '''\
用于通过 URL 查询远程 GraphRAG 服务的轻量级封装。

此类提供了一个简化的接口，用于向已经部署并运行的 GraphRAG 服务进行查询。

Args:
    graphrag_url (str): 远程 GraphRAG 服务端点的基础 URL，应为 'http://hostname:port' 格式。
''')

# rag/graph_retriever.py

add_english_doc('GraphRetriever', '''\
GraphRAG-based retriever for querying knowledge graphs.

This class provides a simple interface for querying GraphRAG knowledge graphs built from Document instances. It acts as a wrapper around GraphDocument's query functionality, providing a consistent retriever interface similar to other retrievers in the LazyLLM framework.

Args:
    doc (Union[Document, GraphDocument]): Either a Document or GraphDocument instance. If a Document is provided, the retriever will attempt to retrieve the associated GraphDocument through a weak reference. If a GraphDocument is provided directly, it will be used as-is.
''')

add_chinese_doc('GraphRetriever', '''\
基于 GraphRAG 的知识图谱查询检索器。

此类提供了一个用于查询由 Document 实例构建的 GraphRAG 知识图谱的简易接口。它封装了 GraphDocument 的查询功能，提供了一个一致的检索器接口，类似于 LazyLLM 框架中的其他检索器。

Args:
    document (Document): Document 或 GraphDocument 实例。如果提供的是 Document，检索器将尝试通过弱引用获取关联的 GraphDocument；如果直接提供 GraphDocument，则按原样使用。
''')

# servers/graphrag/graphrag_server_module.py
add_english_doc('GraphRagServerModule', '''\
GraphRAG server module for managing and operating knowledge graph-based Retrieval Augmented Generation (RAG) services.

This class inherits from ServerModule and provides complete lifecycle management for GraphRAG services.

Args:
    kg_dir (str): Path to the knowledge graph storage directory. The directory will be created automatically if it doesn't exist.
''')

add_chinese_doc('GraphRagServerModule', '''\
GraphRAG 服务器模块用于管理和操作基于知识图谱的检索增强生成 (Retrieval Augmented Generation, RAG) 服务。

该类继承自 ServerModule，并为 GraphRAG 服务提供完整的生命周期管理。

Args:
    kg_dir (str): 知识图谱存储目录的路径。如果该目录不存在，系统将自动创建。
''')


add_english_doc('GraphRagServerModule.prepare_files', '''\
Prepare input files for GraphRAG processing by copying them to the knowledge graph input directory with unique names.

Args:
    files (List[str]): List of file paths to be copied to the input directory. Each path should be a valid file path. Non-existent files will be skipped.
    regenerate_config (bool, optional): Whether to force regeneration of the GraphRAG configuration files. Defaults to True. If True, existing configuration will be overwritten; if False, existing configuration will be preserved if it exists.
''')

add_chinese_doc('GraphRagServerModule.prepare_files', '''\
通过将输入文件复制到知识图谱输入目录并使用唯一名称，来为 GraphRAG 处理准备文件。

Args:
    files (List[str]): 要复制到输入目录的文件路径列表。每个路径都应为有效的文件路径，不存在的文件将被跳过。
    regenerate_config (bool, optional): 是否强制重新生成 GraphRAG 的配置文件。默认值为 True。为 True 时会覆盖已有配置；为 False 时如果已有配置存在则会保留
''')

add_english_doc('GraphRagServerModule.create_index', '''\
Create a knowledge graph index from the prepared input files.

This method sends an asynchronous indexing request to the GraphRAG service. The indexing process runs in the background, and you can check the status using the returned task_id.

Args:
    override (bool, optional): Whether to override an existing index if one already exists. Defaults to True. If False and an index already exists, the request will fail with a 400 error.
''')

add_chinese_doc('GraphRagServerModule.create_index', '''\
从已准备好的输入文件创建知识图谱索引。

该方法会向 GraphRAG 服务发送一个异步索引请求。索引过程在后台运行，可以使用返回的 task_id 查询状态。

Args:
    override (bool, optional): 如果索引已存在，是否覆盖重建。默认值为 True。若为 False 且索引已存在，请求将以 400 错误失败。
''')

add_english_doc('GraphRagServerModule.index_status', '''\
Query the status of an indexing task.

This method retrieves the current status of an indexing task. Use this method to monitor the progress of knowledge graph index creation.

Args:
    task_id (str): The unique identifier of the indexing task, obtained from the create_index() method.
''')

add_chinese_doc('GraphRagServerModule.index_status', '''\
查询构建索引任务的状态。

该方法用于获取某个构建索引任务的当前状态，可用于监控知识图谱索引构建的进度。

Args:
    task_id (str): 索引任务的唯一标识符，来自 create_index() 方法的返回值。
''')

add_english_doc('GraphRagServerModule.query_by_url', '''\
Query a GraphRAG service by URL without requiring a module instance.

This static method allows you to query any GraphRAG service endpoint directly by providing its URL. It's useful when you need to query a remote GraphRAG service or when you don't have a GraphRagServerModule instance available.

Args:
    graphrag_server_url (str): The base URL of the GraphRAG service endpoint. Should be in the format 'http://hostname:port'.
    query (str): The natural language query string to search for in the knowledge graph.
    search_method (str): The search method to use. Can be 'local' (default) or 'global'.
    community_level (int): The community level to use for the search. Defaults to 2.
    response_type (str): The response type to use. Can be 'Multiple Paragraphs' (default) or else.

**Returns:**\n
- dict: A dictionary containing the query result in 'answer' key.
''')

add_chinese_doc('GraphRagServerModule.query_by_url', '''\
使用共享 URL 查询 GraphRAG 知识图谱。

该静态方法允许你通过提供 GraphRAG 服务的 URL，直接对任意 GraphRAG 服务端点进行查询。适用于需要查询远程 GraphRAG 服务，或当前没有可用的 GraphRagServerModule 实例的情况。

Args:
    graphrag_server_url (str): GraphRAG 服务端点的基础 URL，应为 'http://hostname:port' 格式。
    query (str): 用于查询知识图谱的自然语言查询字符串。
    search_method (str): 搜索方式，可为 'local'（默认）或 'global'。
    community_level (int): 搜索使用的社区层级，默认值为 2。
    response_type (str): 响应类型，可为 'Multiple Paragraphs'（默认）或其他类型。
''')

add_english_doc('GraphRagServerModule.query', '''\
Query the GraphRAG service using the instance's service URL.

This method queries the knowledge graph using the instance's service URL.

Args:
    query (str): The natural language query string to search for in the knowledge graph.
    search_method (str): The search method to use. Can be 'local' (default) or 'global'.
    community_level (int): The community level to use for the search. Defaults to 2.
    response_type (str): The response type to use. Can be 'Multiple Paragraphs' (default) or else.

**Returns:**\n
- dict: A dictionary containing the query result in 'answer' key.
''')

add_chinese_doc('GraphRagServerModule.query', '''\
使用该实例的GraphRAG服务 URL 进行问答。

该方法使用实例自身的服务 URL 来查询知识图谱。

Args:
    query (str): 用于查询知识图谱的自然语言查询字符串。
    search_method (str): 搜索方式，可为 'local'（默认）或 'global'。
    community_level (int): 搜索使用的社区层级，默认值为 2。
    response_type (str): 响应类型，可为 'Multiple Paragraphs'（默认）或其他类型。
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
>>> doc = Document(dataset_path="your_files_path")
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
    fill_method (Optional[str]): 缺失值填充策略，可选 'fillna'(默认) / 'ffill' / 'bfill'。
    return_trace (bool): 是否返回处理过程的 trace。
''')

add_english_doc('rag.readers.PandasCSVReader', '''\
Reader for parsing CSV files using pandas.

Args:
    concat_rows (bool): Whether to concatenate all rows into a single text block. Default is True.
    col_joiner (str): String used to join column values.
    row_joiner (str): String used to join rows.
    pandas_config (Optional[Dict]): Optional config for pandas.read_csv.
    fill_method (Optional[str]): Missing value fill strategy: 'fillna'(default) / 'ffill' / 'bfill'.
    return_trace (bool): Whether to return the processing trace.
''')

add_chinese_doc('rag.readers.PandasExcelReader', '''\
用于读取 Excel 文件（.xlsx），并将内容提取为文本。

Args:
    concat_rows (bool): 是否将所有行拼接为一个文本块。
    sheet_name (Optional[str]): 要读取的工作表名称。若为 None，则读取所有工作表。
    pandas_config (Optional[Dict]): pandas.read_excel 的可选配置项。
    fill_method (Optional[str]): 缺失值填充策略，可选 'fillna'(default) / 'ffill' / 'bfill'。
    return_trace (bool): 是否返回处理过程的 trace。
''')

add_english_doc('rag.readers.PandasExcelReader', '''\
Reader for extracting text content from Excel (.xlsx) files.

Args:
    concat_rows (bool): Whether to concatenate all rows into a single block.
    sheet_name (Optional[str]): Name of the sheet to read. If None, all sheets will be read.
    pandas_config (Optional[Dict]): Optional config for pandas.read_excel.
    fill_method (Optional[str]): Missing value fill strategy: 'fillna'(default) / 'ffill' / 'bfill'.
    return_trace (bool): Whether to return the processing trace.
''')

add_chinese_doc('rag.readers.PDFReader', '''\
用于读取 PDF 文件并提取其中的文本内容。

Args:
    split_doc (bool): 若为 True（默认），则解析为一个 `RichDocNode`，可以搭配 `RichTransform` 解析出带有页信息的节点；
        若为 False，则解析为一个纯文本的 `DocNode`。
    post_func (Optional[Callable[[List[DocNode]], List[DocNode]]]): 结果后处理函数，
        需返回 `List[DocNode]`，并会将 `extra_info` 写入每个节点的 `global_metadata`。
    return_trace (bool): 是否返回处理过程的 trace，默认为 True。
    return_full_document (bool, 已弃用): 此参数将在未来版本中删除，请使用 `split_doc` 替代。

Notes:
    当 `split_doc=True` 时返回 `RichDocNode`，否则返回 `DocNode`，两种情况都只返回一个节点。
    当 `split_doc=True` 时，强烈建议搭配 `RichTransform` 使用，可以解析出带有页信息等 metadata 的节点；
    如不使用 `RichTransform`，则解析出的节点会回退为纯文本节点。
''')

add_english_doc('rag.readers.PDFReader', '''\
Reader for extracting text content from PDF files.

Args:
    split_doc (bool): If True (default), parses into a `RichDocNode` which can be used with `RichTransform` to extract nodes with page information;
        if False, parses into a plain text `DocNode`.
    post_func (Optional[Callable[[List[DocNode]], List[DocNode]]]): Post-processing function.
        Must return a ``List[DocNode]`` and will write ``extra_info`` into each node's ``global_metadata``.
    return_trace (bool): Whether to return the processing trace. Default is True.
    return_full_document (bool, deprecated): This parameter will be removed in a future version. Please use `split_doc` instead.

Notes:
    When `split_doc=True`, returns a `RichDocNode`; otherwise returns a `DocNode`. Both cases return a single node.
    When `split_doc=True`, it is strongly recommended to use it with `RichTransform`, which can extract nodes with page information and other metadata;
    without `RichTransform`, the parsed nodes will fall back to plain text nodes.
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
    **kwargs: 其他参数。
''')

add_english_doc('rag.component.bm25.BM25', '''\
A retriever based on the BM25 algorithm that retrieves the most relevant text nodes from a given list of nodes.

Args:
    nodes (List[DocNode]): A list of text nodes to index.
    language (str): The language to use, supports ``en`` (English) and ``zh`` (Chinese). Defaults to ``en``.
    topk (int): The maximum number of nodes to return in each retrieval. Defaults to 2.
    **kwargs: Other parameters.
''')

add_chinese_doc('rag.component.bm25.BM25.retrieve', '''\
使用BM25算法检索与查询最相关的文档节点。

Args:
    query (str): 查询文本。

**Returns:**\n
- List[Tuple[DocNode, float]]: 返回一个列表，每个元素为(文档节点, 相关度分数)的元组。
''')

add_english_doc('rag.component.bm25.BM25.retrieve', '''\
Retrieve the most relevant document nodes for a query using BM25 algorithm.

Args:
    query (str): Query text.

**Returns:**\n
- List[Tuple[DocNode, float]]: Returns a list of tuples containing (document node, relevance score).
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

add_chinese_doc('rag.doc_to_db.DocGenreAnalyser.gen_detection_query', '''\
生成用于文档类型检测的查询。

Args:
    doc_path (str): 文档路径。

**Returns:**\n
- str: 返回格式化的查询字符串，包含文档内容和检测提示。

注意：
    生成的查询会自动根据 ONE_DOC_TOKEN_LIMIT 限制文档内容的长度。
''')

add_english_doc('rag.doc_to_db.DocGenreAnalyser.gen_detection_query', '''\
Generate a query for document type detection.

Args:
    doc_path (str): Path to the document.

**Returns:**\n
- str: Returns a formatted query string containing document content and detection prompts.

Note:
    The generated query will automatically limit document content length based on ONE_DOC_TOKEN_LIMIT.
''')

add_chinese_doc('rag.doc_to_db.DocGenreAnalyser.analyse_doc_genre', '''\
分析文档类型。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 用于分析的语言模型实例。
    doc_path (str): 要分析的文档路径。

**Returns:**\n
- str: 返回检测到的文档类型。如果检测失败则返回空字符串。
''')

add_english_doc('rag.doc_to_db.DocGenreAnalyser.analyse_doc_genre', '''\
Analyze document genre.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): Language model instance for analysis.
    doc_path (str): Path to the document to analyze.

**Returns:**\n
- str: Returns the detected document type. Returns empty string if detection fails.
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

**Returns:**\n
- dict: 提取出的关键信息字典，键为字段名，值为对应的信息值。
''')

add_english_doc('rag.doc_to_db.DocInfoExtractor.extract_doc_info', '''\
Extracts specific key information values from a document according to a provided schema.

This method uses a large language model to analyze document content and extract corresponding information values based on predefined field structure, returning a key-value dictionary.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): The large language model used for document information extraction.
    doc_path (str): Path to the document to be analyzed.
    info_schema (DocInfoSchema): Field structure definition containing the information to be extracted.
    extra_desc (str, optional): Additional description information to guide the extraction process. Defaults to empty string.

**Returns:**\n
- dict: Extracted key information dictionary with field names as keys and corresponding information values as values.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor', '''
基于大模型的结构化信息抽取器：注册 Pydantic schema 后，自动创建/复用数据库表，并将文档内容按字段定义抽取、存储。
可直接用于Document中，文档入库过程中自动生效。

Args:
    db_config (Dict[str, Any]): 目标数据库配置，用于初始化 SqlManager 及建表。
    llm (Union[OnlineChatModule, TrainableModule]): 执行文本抽取的大语言模型。
    table_prefix (str, optional): 自动建表时使用的表名前缀，默认 `lazyllm_schema`。
    force_refresh (bool, optional): 是否强制刷新已有表/缓存。
    extraction_mode (ExtractionMode, optional): 抽取模式，默认为 TEXT，当前仅支持纯文本提取。
    max_len (int, optional): 单文档最大解析长度，默认 102400。
    num_workers (int, optional): 抽取并发线程数，默认 4。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor', '''
LLM-based structured extractor: register a Pydantic schema, auto-create/reuse DB tables, and extract document content into typed fields for persistence.
It can be used into Document, which can make the extracting happen in file parsing progress.

Args:
    db_config (Dict[str, Any]): Database config used to initialize SqlManager and manage tables.
    llm (Union[OnlineChatModule, TrainableModule]): LLM instance used for text extraction.
    table_prefix (str, optional): Prefix for auto-generated tables, defaults to lazyllm_schema.
    force_refresh (bool, optional): Force refresh existing tables/cache.
    extraction_mode (ExtractionMode, optional): Extraction mode, TEXT by default, currently only support text extracting .
    max_len (int, optional): Max length per document to parse, default 102400.
    num_workers (int, optional): Worker threads for extraction, default 4.

''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.register_schema_set', '''
注册 Pydantic schema 集合，必要时创建管理/目标表，返回 schema_set_id（幂等）。

Args:
    schema_set (Type[BaseModel]): 要注册的 Pydantic 模型。
    schema_set_id (str, optional): 自定义 schema 集合 ID，不传则自动生成或复用已有签名。
    force_refresh (bool, optional): 预留参数，期望强制刷新表或缓存时使用。

**Returns:**\n
- str: 注册后的 schema_set_id。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.register_schema_set', '''
Register a Pydantic schema set, creating management/target tables if needed, and return the schema_set_id (idempotent).

Args:
    schema_set (Type[BaseModel]): Pydantic model to register.
    schema_set_id (str, optional): Custom schema set id; auto-generated or reused if omitted.
    force_refresh (bool, optional): Reserved flag for forcing table/cache refresh.

**Returns:**\n
- str: Registered schema_set_id.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.has_schema_set', '''
检查指定 schema_set_id 是否已注册，缺失时会尝试从数据库恢复模型并建表。

Args:
    schema_set_id (str): 目标 schema 集合 ID。

**Returns:**\n
- bool: 是否已存在。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.has_schema_set', '''
Check whether a schema_set_id is registered, recovering the model and ensuring the table if needed.

Args:
    schema_set_id (str): Target schema set id.

**Returns:**\n
- bool: True if it exists.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.register_schema_set_to_kb', '''
将算法/知识库绑定到指定 schema 集合；若提供 schema_set 会先注册；可选 force_refresh 覆盖已有绑定并清理旧数据。

Args:
    algo_id (str, optional): 算法/Document 名称，默认 `__default__`。
    kb_id (str, optional): 知识库 ID，默认 DEFAULT_KB_ID。
    schema_set_id (str, optional): 已有 schema 集合 ID。
    schema_set (Type[BaseModel], optional): 新 schema，传入则会注册后绑定。
    force_refresh (bool, optional): 已绑定不同 schema 时是否强制覆盖并清空旧记录。

**Returns:**\n
- str: 绑定使用的 schema_set_id。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.register_schema_set_to_kb', '''
Bind an algo/kb pair to a schema set; optionally register a provided schema_set first; with force_refresh you can override an existing binding and purge old records.

Args:
    algo_id (str, optional): Algorithm/Document name, defaults to `__default__`.
    kb_id (str, optional): Knowledge base id, defaults to DEFAULT_KB_ID.
    schema_set_id (str, optional): Existing schema set id to bind.
    schema_set (Type[BaseModel], optional): Schema to register and bind if no id is provided.
    force_refresh (bool, optional): Whether to overwrite an existing different binding and clean previous records.

**Returns:**\n
- str: The schema_set_id used for binding.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.analyze_schema_and_register', '''
基于样本文本/DocNode 列表调用大模型推断字段结构，自动生成 Pydantic 模型并注册，返回 SchemaSetInfo（含 schema_set_id 和 pydantic model）。

Args:
    data (Union[str, List[DocNode]]): 用于分析的文本或节点列表（单文档）。
    schema_set_id (str, optional): 自定义/复用的 schema_set_id。

**Returns:**\n
- SchemaSetInfo: 包含 schema_set_id 与生成的 schema 模型。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.analyze_schema_and_register', '''
Infer a schema from sample text or DocNodes using the LLM, auto-create a Pydantic model, register it, and return SchemaSetInfo (id and pydantic model).

Args:
    data (Union[str, List[DocNode]]): Sample text or nodes from a single document.
    schema_set_id (str, optional): Custom or reuse schema_set_id.

**Returns:**\n
- SchemaSetInfo: Contains the schema_set_id and generated schema model.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.extract_and_store', '''
按绑定的 schema 抽取文本/DocNode 内容并写入对应表，若传入 schema_set 会先注册；同文档重复调用会返回缓存结果。

Args:
    data (Union[str, List[DocNode]]): 文本或 DocNode 列表（需同一文档）。
    algo_id (str, optional): 算法/Document 名称，默认 `__default__`。
    schema_set_id (str, optional): 指定使用的 schema 集合 ID。
    schema_set (Type[BaseModel], optional): 动态注册并使用的 schema。

**Returns:**\n
- ExtractResult: 抽取结果，`data` 为字段名到值的字典，`metadata` 包含 schema_set_id、algo_id、kb_id、doc_id 及按字段的线索信息；可能为 None 表示无可写入。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.extract_and_store', '''
Extract content according to the bound schema and persist it; will register the provided schema_set if given; repeated calls for the same doc return cached results.

Args:
    data (Union[str, List[DocNode]]): Text or list of DocNodes from a single document.
    algo_id (str, optional): Algorithm/Document name, defaults to `__default__`.
    schema_set_id (str, optional): Schema set id to use.
    schema_set (Type[BaseModel], optional): Schema to register and use if no id is provided.

**Returns:**\n
- ExtractResult: Result object where `data` is the field/value dict and `metadata` contains schema_set_id, algo_id, kb_id, doc_id, and field-level clues; or None if nothing persisted.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.__call__', '''
便捷调用，等同于 extract_and_store(data, algo_id)，抽取并存储后返回结果。

Args:
    data (Union[str, List[DocNode]]): 文本或 DocNode 列表。
    algo_id (str, optional): 算法/Document 名称。

**Returns:**\n
- ExtractResult: 抽取结果。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.__call__', '''
Convenience wrapper for extract_and_store(data, algo_id), performing extraction and persistence then returning the result.

Args:
    data (Union[str, List[DocNode]]): Text or list of DocNodes.
    algo_id (str, optional): Algorithm/Document name.

**Returns:**\n
- ExtractResult: Extraction result.
''')

add_chinese_doc('rag.doc_to_db.SchemaExtractor.sql_manager_for_nl2sql', '''
基于已绑定的 schema，生成一个仅暴露相关表的 SqlManager，用于 SqlCall 模块中 NL2SQL 查询；会附带表结构描述和可见表列表。

Args:
    algo_id (str, optional): 算法/Document 名称；不传则返回所有绑定关系的可见表。
    kb_ids (Union[str, List[str]], optional): 过滤的知识库 ID，可单个或列表。

**Returns:**\n
- SqlManager: 仅包含可见表、列信息及说明的 SqlManager 实例，用于 NL2SQL。
''')

add_english_doc('rag.doc_to_db.SchemaExtractor.sql_manager_for_nl2sql', '''
Create a SqlManager tailored for NL2SQL in SqlCall Module that only exposes tables bound to the given algo/kb, with descriptions of columns and visible tables.

Args:
    algo_id (str, optional): Algorithm/Document name; when omitted, returns all bound tables.
    kb_ids (Union[str, List[str]], optional): KB id or list to filter bindings.

**Returns:**\n
- SqlManager: Manager instance with visible_tables and column metadata set for NL2SQL use.
''')


add_example('rag.doc_to_db.SchemaExtractor', '''\
from lazyllm.tools.rag import SchemaExtractor
from lazyllm import OnlineChatModule
from pydantic import BaseModel, Field
db_config = {
    "db_type": "sqlite",
    "user": None,
    "password": None,
    "host": None,
    "port": None,
    "db_name": "./test.db",
}
# define a custom pydantic model
class TestSchema(BaseModel):
    company: str = Field(description="Name of the company", default='unknown')
    profit: float = Field(description="Profit of the company, unit is million", default=0.0)

extractor = SchemaExtractor(db_config=db_config, llm=OnlineChatModule(source='siliconflow'), force_refresh=True)
# register to db
extractor.register_schema_set_to_kb(schema_set=TestSchema)
text = "The company name is Apple, and the profit is 100 million."
# you can use it directly by giving a string
res = extractor(data=text)

# bind the schema for a specific algorithm(Document)
extractor.register_schema_set_to_kb(algo_id='algo_1', schema_set=TestSchema)
document = Document(
    dataset_path='./test_docs',
    name="algo_1",
    display_name="Algo_1",
    description="Algo_1 for testing",
    schema_extractor=extractor,  # give it the extractor by this way
)

''')


add_chinese_doc('rag.doc_to_db.DocToDbProcessor', '''\
用于将文档信息抽取并导出到数据库中。

该类通过分析文档主题、抽取字段结构、从文档中提取关键信息，并将其保存至数据库表中。

Args:
    sql_manager (SqlManager): SQL数据库管理器实例
    doc_table_name (str, optional): 文档信息存储表名，默认为"lazyllm_doc_elements"

Note:
    - 如果表已存在，会自动检测并避免重复创建。
    - 如果你希望重置字段结构，使用 `reset_doc_info_schema` 方法。
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor', '''\
Used to extract information from documents and export it to a database.

This class analyzes document topics, extracts schema structure, pulls out key information, and saves it into a database table.

Args:
    sql_manager (SqlManager): SQL database manager instance
    doc_table_name (str, optional): Document information storage table name, defaults to "lazyllm_doc_elements"
Note:
    - If the table already exists, it checks and avoids redundant creation.
    - Use `reset_doc_info_schema` to reset the schema if necessary.
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.extract_info_from_docs', '''\
从文档中提取结构化数据库信息。

该函数使用嵌入和检索技术，在提供的文档中获取数据库相关的文本片段，用于后续模式生成。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 大语言模型实例
    doc_paths (List[str]): 要处理的文档路径列表
    extra_desc (str, optional): 额外的描述信息，用于辅助提取

**Returns:**\n
- List[dict]: 提取的信息字典列表，每个字典对应一个文档的提取结果
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor.extract_info_from_docs', '''\
Extract structured database-related information from documents.

This function uses embedding and retrieval techniques to identify relevant text fragments in the provided documents for schema generation.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): Large language model instance
    doc_paths (List[str]): Document paths to process
    extra_desc (str, optional): Additional description information to assist extraction
**Returns:**\n
- List[dict]: Extracted information dictionary list, each dictionary corresponds to one document's extraction result
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.export_info_to_db', """\
将提取的信息导出到数据库。

将提取的结构化信息批量插入到数据库表中，自动生成UUID和时间戳。

Args:
    info_dicts (List[dict]): 要导出的信息字典列表
""")

add_english_doc('rag.doc_to_db.DocToDbProcessor.export_info_to_db', """\
Export extracted information to database.

Bulk inserts extracted structured information into database table, automatically generating UUID and timestamps.

Args:
    info_dicts (List[dict]): Information dictionary list to export
""")

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.analyze_info_schema_by_llm', '''\
使用大语言模型从文档节点中推断数据库信息结构。

Args:
    llm (Union[OnlineChatModule, TrainableModule]): 大语言模型实例
    doc_paths (List[str]): 文档路径列表
    doc_topic (str, optional): 文档主题，如果为空会自动分析

**Returns:**\n
- DocInfoSchema: 分析得到的文档信息模式列表
''')

add_english_doc('rag.doc_to_db.DocToDbProcessor.analyze_info_schema_by_llm', '''\
Infer structured database information using a large language model from document nodes.

Args:
    llm (Union[OnlineChatModule, TrainableModule]): Large language model instance
    doc_paths (List[str]): Document path list
    doc_topic (str, optional): Document topic, will be automatically analyzed if empty

**Returns:**\n
- DocInfoSchema: Analyzed document information schema list
''')

add_chinese_doc('rag.doc_to_db.DocToDbProcessor.clear', """\
清除处理器状态和数据库表结构。

清空当前文档信息模式、移除ORM类映射，并可选地删除数据库中的文档表。
""")

add_english_doc('rag.doc_to_db.DocToDbProcessor.clear', """\
Clear processor state and database table structures.

Clears current document information schema, removes ORM class mappings, and optionally deletes document table from database.
""")

add_chinese_doc('rag.doc_to_db.extract_db_schema_from_files', '''\
给定文档路径和LLM模型，提取文档结构信息。

Args:
    file_paths (List[str]): 要分析的文档路径。
    llm (Union[OnlineChatModule, TrainableModule]): 支持聊天的模型模块。

**Returns:**\n
- DocInfoSchema: 提取出的字段结构描述。
''')

add_english_doc('rag.doc_to_db.extract_db_schema_from_files', '''\
Extract the schema information from documents using a given LLM.

Args:
    file_paths (List[str]): Paths of the documents to analyze.
    llm (Union[OnlineChatModule, TrainableModule]): A chat-supported LLM module.

**Returns:**\n
- DocInfoSchema: The extracted field structure schema.
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

**Returns:**\n
- List[DocNode]: 包含文档中所有文本内容的节点列表。
""")

add_english_doc('rag.readers.DocxReader', """\
A docx format file parser, reading text content from a `.docx` file and return a list of `DocNode` objects.

Args:
    file (Path): Path to the `.docx` file.
    fs (Optional[AbstractFileSystem]): Optional file system object for custom reading.

**Returns:**\n
- List[DocNode]: A list containing the extracted text content as `DocNode` instances.
""")

add_chinese_doc('rag.readers.EpubReader', """\
用于读取 `.epub` 格式电子书的文件读取器。

继承自 `LazyLLMReaderBase`，只需实现 `_load_data` 方法，即可通过 `Document` 组件自动加载 `.epub` 文件中的内容。

注意：当前版本不支持通过 fsspec 文件系统（如远程路径）加载 epub 文件，若提供 `fs` 参数，将回退到本地文件读取。

**Returns:**\n
- List[DocNode]: 所有章节内容合并后的文本节点列表。
""")

add_english_doc('rag.readers.EpubReader', """\
A file reader for `.epub` format eBooks.

Inherits from `LazyLLMReaderBase`, and only needs to implement `_load_data`. The `Document` module can automatically use this class to load `.epub` files.

Note: Reading from fsspec file systems (e.g., remote paths) is not supported in this version. If `fs` is specified, it will fall back to reading from the local file system.

**Returns:**\n
- List[DocNode]: A single node containing all merged chapter content from the EPUB file.
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
基于Mineru服务的PDF解析器，通过调用Mineru服务的API来解析PDF文件，支持丰富的文档结构识别。

Args:
    url (str): Mineru服务的完整API端点URL。
    backend (str, optional): 解析引擎类型。可选值：
        - 'pipeline': 标准处理流水线
        - 'vlm-transformers': 基于Transformers的视觉语言模型
        - 'vlm-vllm-async-engine': 基于异步VLLM的视觉语言模型
        默认为 'pipeline'。
    upload_mode (bool, optional): 文件传输模式。
        - True: 使用multipart/form-data上传文件内容
        - False: 通过文件路径传递（需确保服务端可访问该路径）
        默认为 False。
    extract_table (bool, optional): 是否提取表格内容并转换为Markdown格式。默认为 True。
    extract_formula (bool, optional): 是否提取公式文本。
        - True: 提取为LaTeX等文本格式
        - False: 将公式保留为图片
        默认为 True。
    split_doc (bool, optional): 若为 True（默认），则解析为一个 `RichDocNode`，可以搭配 `RichTransform` 解析出带有结构信息的节点；
        若为 False，则解析为一个纯文本的 `DocNode`。
    clean_content (bool, optional): 是否清理冗余内容（页眉、页脚、页码等）。默认为 True。
    post_func (Optional[Callable[[List[DocNode]], Any]], optional): 后处理函数，
        接收DocNode列表作为参数，用于自定义结果处理。默认为 None。

Notes:
    当 `split_doc=True` 时返回 `RichDocNode`，否则返回 `DocNode`，两种情况都只返回一个节点。
    当 `split_doc=True` 时，强烈建议搭配 `RichTransform` 使用，可以解析出带有结构信息等 metadata 的节点；
    如不使用 `RichTransform`，则解析出的节点会回退为纯文本节点。
''')

add_english_doc('rag.readers.MineruPDFReader', '''\
Reader for PDF files by calling the Mineru service's API.

Args:
    url (str): The complete API endpoint URL for the Mineru service.
    backend (str, optional): Type of parsing engine. Available options:
        - 'pipeline': Standard processing pipeline
        - 'vlm-transformers': Vision-language model based on Transformers
        - 'vlm-vllm-async-engine': Vision-language model based on async VLLM engine
        Defaults to 'pipeline'.
    upload_mode (bool, optional): File transfer mode.
        - True: Upload file content using multipart/form-data
        - False: Pass by file path (ensure the server can access the path)
        Defaults to False.
    extract_table (bool, optional): Whether to extract table content and convert
        to Markdown format. Defaults to True.
    extract_formula (bool, optional): Whether to extract formula text.
        - True: Extract as text format (e.g., LaTeX)
        - False: Keep formulas as images
        Defaults to True.
    split_doc (bool, optional): If True (default), parses into a `RichDocNode` which can be used with `RichTransform` to extract nodes with structural information;
        if False, parses into a plain text `DocNode`.
    clean_content (bool, optional): Whether to clean redundant content
        (headers, footers, page numbers, etc.). Defaults to True.
    post_func (Optional[Callable[[List[DocNode]], Any]], optional): Post-processing
        function that takes a list of DocNodes as input for custom result handling.
        Defaults to None.

Notes:
    When `split_doc=True`, returns a `RichDocNode`; otherwise returns a `DocNode`. Both cases return a single node.
    When `split_doc=True`, it is strongly recommended to use it with `RichTransform`, which can extract nodes with structural information and other metadata;
    without `RichTransform`, the parsed nodes will fall back to plain text nodes.
''')

add_chinese_doc('rag.readers.MineruPDFReader.set_type_processor', '''\
为特定的内容类型设置自定义处理器函数，用于处理从 Mineru 服务返回的原始内容数据。
返回结果中 'text' 键值将作为 DocNode 的文本内容，其他键值对将作为 DocNode 的元数据（metadata）存储。

Args:
    content_type (str): 内容类型，例如 'text', 'image', 'table', 'equation', 'code', 'list' 等。
    processor (Callable): 处理器函数，接收内容字典作为参数，返回处理后的字典。
''')

add_english_doc('rag.readers.MineruPDFReader.set_type_processor', '''\
Set a custom processor function for a specific content type to process raw content data returned from the Mineru Server.
The 'text' key in the returned dictionary will be used as the DocNode text content, 
while other key-value pairs will be stored as DocNode metadata.

Args:
    content_type (str): Content type, such as 'text', 'image', 'table', 'equation', 'code', 'list' etc.
    processor (Callable): Processor function that takes a dictionary as input and returns a processed dictionary.
''')

add_chinese_doc('rag.readers.PaddleOCRPDFReader', '''\
基于PaddleOCR服务的PDF解析器，通过调用PaddleOCR服务的API来解析PDF文件，支持丰富的文档结构识别。
服务接入方式：
1. 使用官方提供的 API 服务
    - 在飞桨开发者平台（https://aistudio.baidu.com）注册账号并创建 api_key（访问令牌）
    - 初始化时传入 api_key：PaddleOCRPDFReader(api_key="your_api_key")
    - 或者通过环境变量 LAZYLLM_PADDLEOCR_API_KEY 设置 api_key 后初始化：PaddleOCRPDFReader()
2. 使用本地部署的PaddleOCR-VL文档解析服务
    - 服务化部署方式请参考官方文档第 4 节「服务化部署」：https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html
    - 初始化时传入服务地址 url：PaddleOCRPDFReader(url="http://127.0.0.1:8000")

Args:
    url (str, 可选):PaddleOCR 服务的接口地址。如果不提供，使用官方地址。
    api_key (str, 可选):
        访问 PaddleOCR 服务所需的 API Key。 url 与 api_key 至少提供一个。
    format_block_content (bool, 默认 True): 是否将块级内容格式化为 Markdown。
        若需要正确提取多级标题及文档层级结构，必须启用该选项。
    use_layout_detection (bool, 默认 True): 是否启用版面检测进行 PDF 解析。
        若为 False，则每一页仅视为一个整体元素进行处理。
    use_chart_recognition (bool, 默认 True): 是否启用图表识别。
        若为 True，图表将被解析为结构化表格；
        若为 False，图表仅作为普通图片处理。
    split_doc (bool, 默认 True): 若为 True（默认），则解析为一个 `RichDocNode`，可以搭配 `RichTransform` 解析出带有结构信息的节点；
        若为 False，则解析为一个纯文本的 `DocNode`（markdown 内容）。
    drop_types (List[str], 可选): 需要在解析结果中过滤掉的版面块类型列表，
        默认为页眉、页脚、页码、印章等非正文内容。
    post_func (Callable, 可选): 解析完成后对 `DocNode` 列表进行二次处理的后置函数。
        该函数必须接收并返回 `List[DocNode]`。
    images_dir (str, 可选):图片结果的保存目录。
        若提供该参数，解析过程中提取的图片将写入该目录。

Notes:
    当 `split_doc=True` 时返回 `RichDocNode`，否则返回 `DocNode`，两种情况都只返回一个节点。
    当 `split_doc=True` 时，强烈建议搭配 `RichTransform` 使用，可以解析出带有结构信息等 metadata 的节点；
    如不使用 `RichTransform`，则解析出的节点会回退为纯文本节点。
''')

add_english_doc('rag.readers.PaddleOCRPDFReader', '''\
Reader for PDF files by calling the PaddleOCR service's API.
Service Access Methods:
1. Using the official API service
    - Register an account on the PaddlePaddle Developer Platform (https://aistudio.baidu.com) and create an api_key (access token)
    - Pass the api_key during initialization: PaddleOCRPDFReader(api_key="your_api_key")
    - Alternatively, set the api_key via the environment variable LAZYLLM_PADDLEOCR_API_KEY 
      and initialize with: PaddleOCRPDFReader().
2. Using a locally deployed PaddleOCR-VL document parsing service
    - For service-based deployment, refer to Section 4, “Service Deployment,” in the official documentation: 
    https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html
    - Pass the service URL during initialization: PaddleOCRPDFReader(url="http://127.0.0.1:8000")

Args:
    url (str, optional): PaddleOCR service endpoint URL. If not provided, use the official address.
    api_key (str, optional): API key required to access the PaddleOCR service.
        Either url or api_key must be provided.
    format_block_content (bool, default=True): Whether to format block-level content as Markdown.
        This option must be enabled to correctly extract multi-level headings
        and preserve the document's hierarchical structure.
    use_layout_detection (bool, default=True): Whether to enable layout detection during PDF parsing.
        If False, each page is treated as a single, unified element.
    use_chart_recognition (bool, default=True): Whether to enable chart recognition.
        If True, charts are parsed into structured table representations;
        if False, charts are treated as regular images.
    split_doc (bool, default=True): If True (default), parses into a `RichDocNode` which can be used with `RichTransform` to extract nodes with structural information;
        if False, parses into a plain text `DocNode` (containing Markdown text).
    drop_types (List[str], optional): List of layout block types to be excluded from parsing results.
        By default, this includes non-body elements such as headers, footers, page numbers, and seals.
    post_func (Callable, optional): Optional post-processing function applied to the list of `DocNode`
        objects after parsing.
        The function must accept and return a `List[DocNode]`.
    images_dir (str, optional): Directory used to save extracted image results.
        If provided, images extracted during parsing will be written to this directory.

Notes:
    When `split_doc=True`, returns a `RichDocNode`; otherwise returns a `DocNode`. Both cases return a single node.
    When `split_doc=True`, it is strongly recommended to use it with `RichTransform`, which can extract nodes with structural information and other metadata;
    without `RichTransform`, the parsed nodes will fall back to plain text nodes.
''')

add_example('rag.readers.PaddleOCRPDFReader', '''\
from lazyllm.tools.rag.readers import PaddleOCRPDFReader
reader = PaddleOCRPDFReader(url="http://0.0.0.0:9000")  # PaddleOCR server address
nodes = reader("path/to/pdf")
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

**Returns:**\n
- str: 移除图片标签后的内容。
''')

add_english_doc('rag.readers.MarkdownReader.remove_images', '''\
Remove custom image tags of the form ![[...]] from the content.

Args:
    content (str): Input markdown content.

**Returns:**\n
- str: Content with image tags removed.
''')

add_chinese_doc('rag.readers.MarkdownReader.remove_hyperlinks', '''\
移除 Markdown 超链接，将 [文本](链接) 转换为纯文本。

Args:
    content (str): 输入的 markdown 内容。

**Returns:**\n
- str: 移除超链接后的内容，仅保留链接文本。
''')

add_english_doc('rag.readers.MarkdownReader.remove_hyperlinks', '''\
Remove markdown hyperlinks, converting [text](url) to just text.

Args:
    content (str): Input markdown content.

**Returns:**\n
- str: Content with hyperlinks removed, only link text retained.
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


add_english_doc('rag.store.ChromaStore', '''
ChromaStore is a vector-capable implementation of LazyLLMStoreBase, leveraging Chroma for persistence and vector search.

Args:
    uri (Optional[str]): URI string for Chroma connection. Required if `dir` is not provided.
    dir (Optional[str]): Filesystem path for local persistent storage. If provided, PersistentClient mode is used.
    index_kwargs (Optional[Union[Dict, List]]): Configuration for Chroma collections, e.g., index type and distance metrics.
    client_kwargs (Optional[Dict]): Additional arguments passed to the Chroma client constructor.
    **kwargs: Reserved for future extension.
''')

add_chinese_doc('rag.store.ChromaStore', '''
ChromaStore 是基于 Chroma 的向量存储实现，继承自 LazyLLMStoreBase，支持向量写入、检索与持久化。

Args:
    uri (Optional[str]): Chroma 连接 URI，当未指定 `dir` 时必填。
    dir (Optional[str]): 本地持久化存储路径，提供时使用 PersistentClient 模式。
    index_kwargs (Optional[Union[Dict, List]]): Collection 配置参数，如索引类型、距离度量方式等。
    client_kwargs (Optional[Dict]): 传递给 Chroma 客户端的额外参数。
    **kwargs: 预留扩展参数。
''')

add_english_doc('rag.store.ChromaStore.dir', '''
Directory property of the store.

**Returns:**\n
- Optional[str]: Normalized directory path ending with a slash, or None if not set.
''')

add_chinese_doc('rag.store.ChromaStore.dir', '''
存储目录属性。

**Returns:**\n
- Optional[str]: 以斜杠结尾的目录路径，若未配置则返回 None。
''')

add_english_doc('rag.store.ChromaStore.connect', '''
Initialize the Chroma client and configure embedding and global metadata settings.

Args:
    embed_dims (Optional[Dict[str, int]]): Dimensions for each embedding key. Defaults to empty dict if not provided.
    embed_datatypes (Optional[Dict[str, DataType]]): Data types for each embedding key. Only FLOAT_VECTOR or SPARSE_FLOAT_VECTOR are supported.
    global_metadata_desc (Optional[Dict[str, GlobalMetadataDesc]]): Descriptions for global metadata fields. Supported types: string, int, float, bool.
    **kwargs: Reserved for future extension.
''')

add_chinese_doc('rag.store.ChromaStore.connect', '''
初始化 Chroma 客户端并配置向量化及元数据相关设定。

Args:
    embed_dims (Optional[Dict[str, int]]): 每个嵌入键对应的向量维度，未提供时默认为空字典。
    embed_datatypes (Optional[Dict[str, DataType]]): 每个嵌入键的数据类型，仅支持 FLOAT_VECTOR 或 SPARSE_FLOAT_VECTOR。
    global_metadata_desc (Optional[Dict[str, GlobalMetadataDesc]]): 全局元数据字段的描述，支持类型：字符串、整型、浮点型、布尔型。
    **kwargs: 预留扩展参数。
''')

add_english_doc('rag.store.ChromaStore.upsert', '''
Insert or update a batch of records(segment's uid and vectors) into Chroma.

Args:
    collection_name (str): Logical name for the collection.
    data (List[dict]): List of documents.

**Returns:**\n
- bool: True if operation succeeds, False otherwise.
''')

add_chinese_doc('rag.store.ChromaStore.upsert', '''
批量写入或更新记录（切片的id及向量数据）到 Chroma。

Args:
    collection_name (str): 集合名称。
    data (List[dict]): 文档切片数据列表。

**Returns:**\n
- bool: 操作成功返回 True，否则 False。
''')

add_english_doc('rag.store.ChromaStore.delete', '''
Delete an entire collection or specific records.

Args:
    collection_name (str): Name of the collection to delete from.
    criteria (Optional[dict]): If None, delete the entire collection. Otherwise, a dictionary specifying conditions to delete matching records (e.g., by doc_id, uid, kb_id).
    **kwargs: Reserved for future extension.

**Returns:**\n
- bool: True if deletion succeeds, False otherwise.
''')

add_chinese_doc('rag.store.ChromaStore.delete', '''
删除整个集合或指定记录。

Args:
    collection_name (str): 要删除的集合名称。
    criteria (Optional[dict]): 若为 None，则删除整个集合；否则按字典条件删除匹配的记录（例如按 doc_id、uid、kb_id 删除）。
    **kwargs: 预留扩展参数。

**Returns:**\n
- bool: 删除成功返回 True，否则返回 False。
''')

add_english_doc('rag.store.ChromaStore.get', '''
Retrieve records matching criteria.

Args:
    collection_name (str): Name of the collection to query.
    criteria (Optional[dict]): Filter conditions such as primary key or metadata (e.g., doc_id, kb_id). If None, retrieves all records.

**Returns:**\n
- List[dict]: A list of records, where each record contains:
    - 'uid': The unique identifier of the record.
    - 'global_meta': A dictionary of global metadata fields.
    - 'embedding': A dictionary mapping embedding keys to their corresponding vectors.
''')

add_chinese_doc('rag.store.ChromaStore.get', '''
根据条件检索记录。

Args:
    collection_name (str): 要查询的集合名称。
    criteria (Optional[dict]): 过滤条件，如主键或元数据（例如 doc_id、kb_id）。若为 None，则返回集合中所有记录。

**Returns:**\n
- List[dict]: 记录列表，每条记录包含：
    - 'uid': 记录的唯一标识符。
    - 'global_meta': 全局元数据字段的字典。
    - 'embedding': 嵌入键到对应向量的映射。
''')

add_english_doc('rag.store.ChromaStore.search', '''
Perform a vector similarity search.

Args:
    collection_name (str): Name of the collection to query.
    query_embedding (List[float]): The query vector for similarity search.
    embed_key (str): The embedding key specifying which embedding space to use.
    topk (int, optional): Number of top results to return. Defaults to 10.
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): Optional metadata filter conditions to restrict search results.

**Returns:**\n
- List[dict]: A list of matched records, where each record contains:
    - 'uid': The unique identifier of the matched record.
    - 'score': The similarity score (1 - distance).
''')

add_chinese_doc('rag.store.ChromaStore.search', '''
执行向量相似度检索。

Args:
    collection_name (str): 要查询的集合名称。
    query_embedding (List[float]): 用于检索的向量。
    embed_key (str): 指定使用的向量空间 key。
    topk (int, optional): 返回的结果数量，默认为 10。
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): 可选的元数据过滤条件，用于限制检索结果。

**Returns:**\n
- List[dict]: 匹配结果列表，每条记录包含：
    - 'uid': 匹配记录的唯一标识符。
    - 'score': 相似度分数（1 - 距离）。
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

**Returns:**\n
- Optional[str]: Directory path for local milvus.db file, or None if remote.
''')

add_chinese_doc('rag.store.MilvusStore.dir', '''
存储目录属性，基于 URI 推断。远程模式返回 None。

**Returns:**\n
- Optional[str]: 本地 milvus.db 文件的目录路径，或 None。
''')

add_english_doc('rag.store.MilvusStore.connect', '''
Initialize Milvus client, pass in embedding model parameters and global metadata descriptions.

Args:
    embed_dims (Dict[str, int]): Embedding dimensions per embed key.
    embed_datatypes (Dict[str, DataType]): Data types for each embed key.
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Descriptions for metadata fields.
    kwargs: Other connection parameters
''')

add_chinese_doc('rag.store.MilvusStore.connect', '''
初始化 Milvus 客户端，传入向量化模型参数和全局元数据描述。

Args:
    embed_dims (Dict[str, int]): 每个嵌入键对应的向量维度。
    embed_datatypes (Dict[str, DataType]): 每个嵌入键的数据类型。
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): 全局元数据字段的描述。
    kwargs: 其他连接参数
''')

add_english_doc('rag.store.MilvusStore.upsert', '''
Insert or update a batch of segment data into the Milvus collection.

Args:
    collection_name (str): Collection name (per embed key grouping).
    data (List[dict]): List of segment data.

**Returns:**\n
- bool: True if successful, False otherwise.
''')

add_chinese_doc('rag.store.MilvusStore.upsert', '''
批量写入或更新切片数据到 Milvus 集合。

Args:
    collection_name (str): 集合名称，通常为 "group_embedKey" 格式。
    data (List[dict]): 切片数据列表。

**Returns:**\n
- bool: 操作成功返回 True，否则 False。
''')

add_english_doc('rag.store.MilvusStore.delete', '''
Delete entire collection or subset of records by criteria.

Args:
    collection_name (str): Target collection.
    criteria (Optional[dict]): If None, drop the entire collection; otherwise a dict of filters (uid list or metadata conditions).
    kwargs: Other delete parameters

**Returns:**\n
- bool: True if deletion succeeds, False otherwise.
''')

add_chinese_doc('rag.store.MilvusStore.delete', '''
删除整个集合或按条件删除指定记录。

Args:
    collection_name (str): 目标集合名称。
    criteria (Optional[dict]): 若为 None 则删除整个集合；否则按 uid 列表或元数据条件过滤。
    kwargs: 其他查询参数

**Returns:**\n
- bool: 如果删除成功返回True，否则返回False。
''')

add_english_doc('rag.store.MilvusStore.get', '''
Retrieve records matching primary-key or metadata filters.

Args:
    collection_name (str): Collection to query.
    criteria (Optional[dict]): Dict containing 'uid' list or metadata field filters.
    kwargs: Other query parameters

**Returns:**\n
- List[dict]: Each entry contains 'uid' and 'embedding'.
''')

add_chinese_doc('rag.store.MilvusStore.get', '''
检索匹配主键或元数据过滤条件的记录。

Args:
    collection_name (str): 待查询集合。
    criteria (Optional[dict]): 包含 'uid' 列表或元数据字段过滤条件。
    kwargs: 其他查询参数

**Returns:**\n
- List[dict]: 每项包含 'uid' 及 'embedding' 映射。
''')

add_english_doc('rag.store.MilvusStore.search', '''
Perform vector similarity search with optional metadata filtering.

Args:
    collection_name (str): Collection to search.
    query_embedding (List[float]): Query vector.
    topk (int): Number of nearest neighbors.
    filters (Optional[Dict[str, Union[List, Set]]]): Metadata filter map.
    embed_key (str): Which embedding field to use.
    filter_str (Optional[str], optional): 过滤表达式字符串。默认为空字符串
    kwargs: 其他搜索参数

**Returns:**\n
- List[dict]: Each dict has 'uid' and similarity 'score'.
''')

add_chinese_doc('rag.store.MilvusStore.search', '''
执行向量相似度检索，并可按元数据过滤。

Args:
    collection_name (str): 待搜索集合。
    query_embedding (List[float]): 查询向量。
    topk (int): 返回邻近数量。
    filters (Optional[Dict[str, Union[List, Set]]]): 元数据过滤映射。
    embed_key (str): 使用的嵌入字段。
    filter_str (Optional[str], optional): Filter expression string. Defaults to empty string
    kwargs: Other search parameters

**Returns:**\n
- List[dict]: 每项包含 'uid' 及相似度 'score'。
''')

add_english_doc('rag.store.ElasticSearchStore', '''
Vector store implementation based on Elasticsearch, inheriting from StoreBase. Supports vector insertion, deletion, flexible querying (including scalar filtering).
Args:
    uris (List[str]): Elasticsearch connection URIs (e.g., ["http://localhost:9200"]).
    client_kwargs (Optional[Dict]): Additional keyword arguments for Elasticsearch client.
    index_kwargs (Optional[Union[Dict, List]]): Index creation parameters (e.g., {"index_type": "IVF_FLAT", "metric_type": "COSINE"} or a list of per-embed-key configs).
    **kwargs: Additional keyword arguments.
''')

add_chinese_doc('rag.store.ElasticSearchStore', '''
基于 Elasticsearch 的向量存储实现，继承自 StoreBase。支持向量写入、删除、相似度检索，兼容标量过滤。
Args:
    uris (List[str]): Elasticsearch 连接 URI（如 ["http://localhost:9200"]）。
    client_kwargs (Optional[Dict]): 传递给 Elasticsearch 客户端的额外参数。
    index_kwargs (Optional[Union[Dict, List]]): 索引创建参数（例如 {"index_type": "IVF_FLAT", "metric_type": "CONSINE"} ，支持按向量模型的key配置列表）。
    **kwargs: 预留扩展参数。
''')

add_example('rag.store.ElasticSearchStore', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.store import ElasticSearchStore
>>> store = ElasticSearchStore(uris=["localhost:9200"], client_kwargs={}, index_kwargs={})
>>> store.connect(embed_dims={"vec_dense": 128, "vec_sparse": 128}, embed_datatypes={"vec_dense": DataType.FLOAT32, "vec_sparse": DataType.FLOAT32}, global_metadata_desc={})
>>> store.upsert(collection_name="test", data=[{"uid": "1", "embedding": {"vec_dense": [0.1, 0.2, 0.3], "vec_sparse": {"1": 0.1, "2": 0.2, "3": 0.3}}, "metadata": {"key1": "value1", "key2": "value2"}}])
>>> store.get(collection_name="test", criteria={"uid": "1"})
>>> store.delete(collection_name="test", criteria={"uid": "1"})
''')

add_english_doc('rag.store.ElasticSearchStore.dir', '''
Returns None when using remote Elasticsearch.
**Returns:**\n
    Optional[str]: None if remote.
''')

add_chinese_doc('rag.store.ElasticSearchStore.dir', '''
远程模式返回 None。
**Returns:**\n
    Optional[str]: None。
''')

add_english_doc('rag.store.ElasticSearchStore.connect', '''
Initialize Elasticsearch client, pass in embedding model parameters and global metadata descriptions.
Args:
    embed_dims (Dict[str, int]): Embedding dimensions per embed key.
    embed_datatypes (Dict[str, DataType]): Data types for each embed key.
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Descriptions for metadata fields.
**Returns:**\n
    bool: True if successful, False otherwise.
''')

add_chinese_doc('rag.store.ElasticSearchStore.connect', '''
初始化 Elasticsearch 客户端，传入向量化模型参数和全局元数据描述。
Args:
    embed_dims (Dict[str, int]): 每个嵌入键对应的向量维度。
    embed_datatypes (Dict[str, DataType]): 每个嵌入键的数据类型。
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): 全局元数据字段的描述。
**Returns:**\n
    bool: 操作成功返回 True，否则 False。
''')

add_english_doc('rag.store.ElasticSearchStore.upsert', '''
Insert or update a batch of segment data into the Elasticsearch collection.
Args:
    collection_name (str): Collection name (per embed key grouping).
    data (List[dict]): List of segment data.
**Returns:**\n
    bool: True if successful, False otherwise.
''')

add_chinese_doc('rag.store.ElasticSearchStore.upsert', '''
批量写入或更新切片数据到 Elasticsearch 集合。
Args:
    collection_name (str): 集合名称，通常为 "group_embedKey" 格式。
    data (List[dict]): 切片数据列表。
**Returns:**\n
    bool: 操作成功返回 True，否则 False。
''')

add_english_doc('rag.store.ElasticSearchStore.delete', '''
Delete entire collection or subset of records by criteria.
Args:
    collection_name (str): Target collection.
    criteria (Optional[dict]): If None, drop the entire collection; otherwise a dict of filters (uid list or metadata conditions).
**Returns:**\n
    bool: True if deletion succeeds, False otherwise.
''')

add_chinese_doc('rag.store.ElasticSearchStore.delete', '''
删除整个集合或按条件删除指定记录。
Args:
    collection_name (str): 目标集合名称。
    criteria (Optional[dict]): 若为 None 则删除整个集合；否则按 uid 列表或元数据条件过滤。
**Returns:**\n
    bool: 删除成功返回 True，否则 False。
''')

add_english_doc('rag.store.ElasticSearchStore.get', '''
Retrieve records matching primary-key or metadata filters.
Args:
    collection_name (str): Collection to query.
    criteria (Optional[dict]): Dict containing 'uid' list or metadata field filters.
**Returns:**\n
    List[dict]: List of segments with 'uid' and 'embedding'.
''')

add_chinese_doc('rag.store.ElasticSearchStore.get', '''
检索匹配主键或元数据过滤条件的记录。
Args:
    collection_name (str): 待查询集合。
    criteria (Optional[dict]): 包含 'uid' 列表或元数据字段过滤条件。
**Returns:**\n
    List[dict]: 每项包含 'uid' 及 'embedding' 映射。
''')

add_english_doc('rag.store.ElasticSearchStore.search', '''
Perform vector similarity search with optional metadata filtering.
Args:
    collection_name (str): Collection to search.
    query (Optional[str]): Query string.
    topk (Optional[int]): Number of nearest neighbors.
    filters (Optional[dict]): Metadata filter map.
    kwargs: Other search parameters

**Returns:**\n
- List[dict]: Return matching results list and similarity 'score'.
''')

add_chinese_doc('rag.store.ElasticSearchStore.search', '''
执行向量相似度检索，并可按元数据过滤。
Args:
    collection_name (str): 待搜索集合。
    query (Optional[str]): 查询字符串。
    topk (Optional[int]): 返回邻近数量。
    filters (Optional[dict]): 元数据过滤映射。
    kwargs: 其他搜索参数

**Returns:**\n
- List[dict]: 返回匹配结果列表及相似度 'score'。
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore', '''\
混合存储类，结合了分段存储和向量存储的功能。

Args:
    segment_store (LazyLLMStoreBase): 分段存储实例，用于存储文档的原始内容。
    vector_store (LazyLLMStoreBase): 向量存储实例，用于存储文档的向量表示。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore', '''\
Hybrid storage class that combines segment storage and vector storage capabilities.

Args:
    segment_store (LazyLLMStoreBase): Segment storage instance for storing original document content.
    vector_store (LazyLLMStoreBase): Vector storage instance for storing document vector representations.
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore.connect', '''\
连接到底层的分段存储和向量存储。

Args:
    *args: 传递给存储连接方法的位置参数。
    **kwargs: 传递给存储连接方法的关键字参数。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore.connect', '''\
Connect to underlying segment and vector stores.

Args:
    *args: Positional arguments passed to store connection methods.
    **kwargs: Keyword arguments passed to store connection methods.
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore.upsert', '''\
向存储中插入或更新数据。

Args:
    collection_name (str): 集合名称。
    data (List[dict]): 要插入或更新的数据列表，每个数据项都是一个字典。

**Returns:**\n
- bool: 操作成功返回True，否则返回False。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore.upsert', '''\
Insert or update data in the stores.

Args:
    collection_name (str): Name of the collection.
    data (List[dict]): List of data items to insert or update, each item is a dictionary.

**Returns:**\n
- bool: Returns True if operation is successful, False otherwise.
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore.delete', '''\
从存储中删除数据。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 删除条件，默认为None。
    **kwargs: 其他参数。

**Returns:**\n
- bool: 操作成功返回True，否则返回False。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore.delete', '''\
Delete data from the stores.

Args:
    collection_name (str): Name of the collection.
    criteria (Optional[dict]): Deletion criteria, defaults to None.
    **kwargs: Additional arguments.

**Returns:**\n
- bool: Returns True if operation is successful, False otherwise.
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore.get', '''\
从存储中获取数据。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 查询条件，默认为None。
    **kwargs: 其他参数。

**Returns:**\n
- List[dict]: 返回符合条件的数据列表。

Raises:
    ValueError: 当向量存储中的uid在分段存储中找不到时抛出。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore.get', '''\
Retrieve data from the stores.

Args:
    collection_name (str): Name of the collection.
    criteria (Optional[dict]): Query criteria, defaults to None.
    **kwargs: Additional arguments.

**Returns:**\n
- List[dict]: List of matching data items.

Raises:
    ValueError: When a uid found in vector store is not found in segment store.
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore.search', '''\
在存储中搜索数据。

Args:
    collection_name (str): 集合名称。
    query (str): 搜索查询字符串。
    query_embedding (Optional[Union[dict, List[float]]]): 查询的向量表示，默认为None。
    topk (int): 返回的最大结果数量，默认为10。
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): 过滤条件，默认为None。
    embed_key (Optional[str]): 嵌入向量的键名，默认为None。
    **kwargs: 其他参数。

**Returns:**\n
- List[dict]: 返回搜索结果列表。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore.search', '''\
Search data in the stores.

Args:
    collection_name (str): Name of the collection.
    query (str): Search query string.
    query_embedding (Optional[Union[dict, List[float]]]): Vector representation of the query, defaults to None.
    topk (int): Maximum number of results to return, defaults to 10.
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): Filter conditions, defaults to None.
    embed_key (Optional[str]): Key name for embedding vector, defaults to None.
    **kwargs: Additional arguments.

**Returns:**\n
- List[dict]: List of search results.
''')

add_chinese_doc('rag.store.hybrid.hybrid_store.HybridStore.drop_collection', '''\
删除指定集合，同时从分段存储和向量存储中移除对应数据。

Args:
    collection_name (str): 要删除的集合名称。

**Returns:**\n
- bool: 若两个底层存储均成功删除则返回 ``True``，任意一个失败则返回 ``False``。
''')

add_english_doc('rag.store.hybrid.hybrid_store.HybridStore.drop_collection', '''\
Drop a collection from both the segment store and the vector store.

Args:
    collection_name (str): Name of the collection to drop.

**Returns:**\n
- bool: ``True`` if both underlying stores dropped the collection successfully, ``False`` otherwise.
''')

add_chinese_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore', '''\
OceanBase 存储类，用于存储和检索文档节点。

Args:
    uri (str): OceanBase 数据库的 URI。
    user (str): OceanBase 数据库的用户名。
    password (str): OceanBase 数据库的密码。
    db_name (str): OceanBase 数据库的名称。
    drop_old (bool): 是否删除旧的表。
    index_kwargs (List[dict]): 索引配置列表。
    client_kwargs (Dict): 客户端配置字典。
    max_pool_size (int): 最大连接池大小。
    normalize (bool): 是否规范化数据。
    enable_fulltext_index (bool): 是否启用全文索引。
''')

add_english_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore', '''\
OceanBase storage class for storing and retrieving document nodes.

Args:
    uri (str): URI of the OceanBase database.
    user (str): Username of the OceanBase database.
    password (str): Password of the OceanBase database.
    db_name (str): Name of the OceanBase database.
    drop_old (bool): Whether to drop old tables.
    index_kwargs (List[dict]): List of index configurations.
    client_kwargs (Dict): Dictionary of client configurations.
    max_pool_size (int): Maximum pool size.
    normalize (bool): Whether to normalize data.
    enable_fulltext_index (bool): Whether to enable fulltext index.
''')

add_chinese_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.connect', '''\
连接到底层的 OceanBase 数据库。

Args:
    embed_dims (Dict[str, int]): 嵌入维度字典。
    embed_datatypes (Dict[str, DataType]): 嵌入数据类型字典。
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): 全局元数据描述字典。
    **kwargs: 其他参数。
''')

add_english_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.connect', '''\
Connect to underlying OceanBase database.

Args:
    embed_dims (Dict[str, int]): Dictionary of embedding dimensions.
    embed_datatypes (Dict[str, DataType]): Dictionary of embedding data types.
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Dictionary of global metadata descriptions.
    **kwargs: Additional arguments.
''')

add_chinese_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.upsert', '''\
向存储中插入或更新数据。

Args:
    collection_name (str): 集合名称。
    data (List[dict]): 要插入或更新的数据列表，每个数据项都是一个字典。
    range_part (Optional[RangeListPartInfo]): 范围分区信息，暂未实现分区功能。
    **kwargs: 其他参数。

**Returns:**\n
- bool: 操作成功返回True，否则返回False。
''')

add_english_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.upsert', '''\
Insert or update data in the stores.

Args:
    collection_name (str): Name of the collection.
    data (List[dict]): List of data items to insert or update, each item is a dictionary.
    range_part (Optional[RangeListPartInfo]): Range partition information, not implemented yet.
    **kwargs: Additional arguments.

**Returns:**\n
- bool: Returns True if operation is successful, False otherwise.
''')

add_chinese_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.delete', '''\
从存储中删除数据。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 删除条件，默认为None。
    **kwargs: 其他参数。

**Returns:**\n
- bool: 操作成功返回True，否则返回False。
''')

add_english_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.delete', '''\
Delete data from the stores.

Args:
    collection_name (str): Name of the collection.
    criteria (Optional[dict]): Deletion criteria, defaults to None.
    **kwargs: Additional arguments.

**Returns:**\n
- bool: Returns True if operation is successful, False otherwise.
''')

add_chinese_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.get', '''\
从存储中获取数据。

Args:
    collection_name (str): 集合名称。
    criteria (Optional[dict]): 查询条件，默认为None。
    **kwargs: 其他参数。

**Returns:**\n
- List[dict]: 返回符合条件的数据列表。
''')

add_english_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.get', '''\
Retrieve data from the stores.

Args:
    collection_name (str): Name of the collection.
    criteria (Optional[dict]): Query criteria, defaults to None.
    **kwargs: Additional arguments.

**Returns:**\n
- List[dict]: List of matching data items.
''')

add_chinese_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.search', '''\
在存储中搜索数据。

Args:
    collection_name (str): 集合名称。
    query_embedding (Union[dict, List[float]]): 查询的向量表示。
    topk (int): 返回的最大结果数量。
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): 过滤条件，默认为None。
    embed_key (Optional[str]): 嵌入向量的键名，默认为None。
    filter_str (Optional[str]): 过滤条件字符串，默认为None。
    **kwargs: 其他参数。

**Returns:**\n
- List[dict]: 返回搜索结果列表。
''')

add_english_doc('rag.store.hybrid.oceanbase_store.OceanBaseStore.search', '''\
Search data in the stores.

Args:
    collection_name (str): Name of the collection.
    query_embedding (Union[dict, List[float]]): Vector representation of the query.
    topk (int): Maximum number of results to return.
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): Filter conditions, defaults to None.
    embed_key (Optional[str]): Key name for embedding vector, defaults to None.
    filter_str (Optional[str]): Filter conditions string, defaults to None.
    **kwargs: Additional arguments.

**Returns:**\n
- List[dict]: List of search results.
''')

add_chinese_doc('rag.store.hybrid.sensecore_store.SenseCoreStore', '''\
SenseCore 混合存储实现，继承自 LazyLLMStoreBase，提供基于 SenseCore 平台的文档存储和检索功能。
该类支持文档的序列化存储、多模态内容处理、混合搜索等功能，通过 S3 存储和 SenseCore API 实现高效的文档管理。

功能特性:
    - 支持全功能存储能力（StoreCapability.ALL），包括插入、删除、查询、搜索等操作。
    - 自动处理图像内容，将本地图像上传到 S3 存储并生成访问链接。
    - 支持多模态搜索，包括文本和图像混合查询。
    - 提供文档序列化和反序列化功能，支持复杂数据结构存储。
    - 支持批量操作和异步任务处理，提高存储效率。
    - 集成 S3 存储和 SenseCore API，实现云端文档管理。

Args:
    uri (str): SenseCore 服务的 API 地址，默认为空字符串。
    **kwargs: 其他配置参数，包括 s3_config 和 image_url_config。

配置参数:
    s3_config (dict): S3 存储配置，包含 bucket_name、access_key、secret_access_key 等。
    image_url_config (dict): 图像 URL 生成配置，用于多模态搜索。

''')

add_english_doc('rag.store.hybrid.sensecore_store.SenseCoreStore', '''\
SenseCore hybrid storage implementation, inheriting from LazyLLMStoreBase, providing document storage and retrieval functionality based on the SenseCore platform.
This class supports document serialization storage, multimodal content processing, hybrid search, and other features, implementing efficient document management through S3 storage and SenseCore API.

Key Features:
    - Supports full storage capabilities (StoreCapability.ALL), including insert, delete, query, search operations.
    - Automatically handles image content, uploading local images to S3 storage and generating access links.
    - Supports multimodal search, including text and image hybrid queries.
    - Provides document serialization and deserialization functionality, supporting complex data structure storage.
    - Supports batch operations and asynchronous task processing for improved storage efficiency.
    - Integrates S3 storage and SenseCore API for cloud-based document management.

Args:
    uri (str): SenseCore service API address, defaults to empty string.
    **kwargs: Additional configuration parameters, including s3_config and image_url_config.

Configuration Parameters:
    s3_config (dict): S3 storage configuration, including bucket_name, access_key, secret_access_key, etc.
    image_url_config (dict): Image URL generation configuration for multimodal search.

''')

add_chinese_doc('rag.default_index.DefaultIndex', '''\
默认的索引实现，负责通过 embedding 和文本相似度在底层存储中查询、更新和删除文档节点。支持多种相似度度量方式，并在必要时对查询和节点进行 embedding 计算与更新。

Args:
    embed (Dict[str, Callable]): 用于生成查询和节点 embedding 的字典，key 是 embedding 名称，value 是接收字符串返回向量的函数。
    store (StoreBase): 底层存储，用于持久化和检索 DocNode 节点。
    **kwargs: 预留扩展参数。

**Returns:**\n
- DefaultIndex: 默认索引实例。
''')

add_english_doc('rag.default_index.DefaultIndex', '''\
Default index implementation responsible for querying, updating, and removing document nodes in the underlying store based on embedding or text similarity.
Supports multiple similarity metrics and performs embedding computation and node updates when required.

Args:
    embed (Dict[str, Callable]): Mapping of embedding names to functions that generate vector representations from strings.
    store (StoreBase): Underlying storage to persist and retrieve `DocNode` objects.
    **kwargs: Reserved for future extension.

**Returns:**\n
- DefaultIndex: The default index instance.
''')

add_chinese_doc('rag.default_index.DefaultIndex.update', '''\
根据提供的节点列表更新索引中的内容。具体行为由子类或外部实现填充（此处为空实现，需在实际使用中覆盖/扩展）。

Args:
    nodes (List[DocNode]): 需要更新（新增或替换）的文档节点列表。
''')

add_english_doc('rag.default_index.DefaultIndex.update', '''\
Update the index with the given list of document nodes. This is a placeholder implementation and should be provided/extended in concrete usage.

Args:
    nodes (List[DocNode]): Document nodes to add or update in the index.
''')

add_chinese_doc('rag.default_index.DefaultIndex.remove', '''\
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

add_chinese_doc('rag.default_index.DefaultIndex.query', '''\
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

**Returns:**\n
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

add_english_doc('rag.retriever.TempRetriever', '''
TempRetriever Base class. used for `TempDocRetriever` and `ContextRetriever`.

Args:
    embed: The embedding function.
    output_format: The format of the output result (e.g., JSON). Optional, defaults to None.
    join: Whether to merge multiple result segments (set to True or specify a separator like "\n").
''')

add_chinese_doc('rag.retriever.TempRetriever', '''
临时文档检索器基类，用于 `TempDocRetriever` 和 `ContextRetriever`。

Args:
    embed:嵌入函数。
    output_format:结果输出格式(如json),可选默认为None
    join:是否合并多段结果(True或用分隔符如"\n")
''')

add_english_doc('rag.retriever.TempDocRetriever', '''
A temporary document retriever that inherits from TempRetriever, used for quickly processing temporary files and performing retrieval tasks.

Args:
    embed: The embedding function.
    output_format: The format of the output result (e.g., JSON). Optional, defaults to None.
    join: Whether to merge multiple result segments (set to True or specify a separator like "\n").
''')

add_chinese_doc('rag.retriever.TempDocRetriever', '''
临时文档检索器，继承自TempRetriever，用于快速处理临时文件并执行检索任务。

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
    files = ["/path/to/file.txt"]
    results = retriever.forward(files, "什么是机器学习?")
    print(results)
''')

add_english_doc('rag.retriever.TempRetriever.create_node_group', '''
Create document processing node group for configuring document chunking and transformation strategies.

Args:
    name (str): Name of the node group. Auto-generated if None.
    transform (Callable): Function to process documents in this group.
    parent (str): Parent group name. Defaults to root group.
    trans_node (bool): Whether to transform nodes. Inherits from parent if None.
    num_workers (int): Parallel workers for processing. Default 0 (sequential).
    **kwargs: Additional group parameters.

**Returns:**\n
- self: Current instance supporting chained calls
''')

add_english_doc('rag.retriever.ContextRetriever', '''
A context-based retriever that inherits from TempRetriever, designed to perform retrieval directly over in-memory text content rather than physical document files.

It internally converts the provided context strings into temporary files using TempPathGenerator, builds retrievers on demand, and caches them for efficient reuse.

Args:
    embed: The embedding function used for vector-based retrieval. If not provided, a keyword-based method (e.g., BM25) is used.
    output_format: The format of the output result (e.g., "text", "json"). Optional, defaults to None.
    join: Whether to merge multiple retrieved segments. Can be True or a custom separator string such as "\\n".
''')

add_chinese_doc('rag.retriever.ContextRetriever', '''
基于上下文内容的检索器，继承自 TempRetriever，用于直接对内存中的文本内容进行检索，而非依赖真实存在的文档文件。

该检索器会通过 TempPathGenerator 将传入的上下文字符串临时转换为文件路径，
在此基础上构建 Retriever，并使用 LRU 缓存以提升重复查询时的性能。

Args:
    embed: 用于向量检索的嵌入函数；若未提供，则自动退化为关键词检索（如 BM25）。
    output_format: 结果输出格式（如 "text"、"json"），可选，默认 None。
    join: 是否合并多段检索结果，可为 True 或自定义分隔符（如 "\\n"）。
''')

add_example('rag.retriever.ContextRetriever', '''\
>>> ctx1 = '大学之道，在明明德，\n在亲民，在止于至善。\n知止而后有定，定而后能静，静而后能安。'
>>> ctx2 = '子曰：学而时习之，不亦说乎？\n有朋自远方来，不亦乐乎？'
>>> ret = ContextRetriever(output_format='dict')
>>> ret.create_node_group('block', transform=lambda x: x.split('\n'))
>>> ret.add_subretriever(Document.CoarseChunk, topk=1)
>>> ret.add_subretriever('block', topk=3)
>>> ret([ctx1, ctx2], '大学')
''')


add_chinese_doc('rag.retriever.TempRetriever.create_node_group', '''
创建文档处理节点组，用于配置文档的分块和转换策略。

Args:
    name (str): 节点组名称，None时自动生成。
    transform (Callable): 该组文档的处理函数。
    parent (str): 父组名称，默认为根组。
    trans_node (bool): 是否转换节点，None时继承父组设置。
    num_workers (int): 并行处理worker数，0表示串行。
    **kwargs: 其他组参数。

**Returns:**\n
- self: 支持链式调用的当前实例
''')

add_english_doc('rag.retriever.TempRetriever.add_subretriever', '''
Add a sub-retriever with search configuration.

Args:
    group (str): Target node group name.
    **kwargs: Retriever configuration parameters including:
        - similarity (str): Similarity calculation method, 'cosine' (cosine similarity) or 'bm25' (BM25 algorithm)
        - Other retriever-specific parameters

**Returns:**\n
- self: For method chaining.
''')

add_chinese_doc('rag.retriever.TempRetriever.add_subretriever', '''
添加带搜索配置的子检索器。

Args:
    group (str): 节点组名称，指定使用哪个已配置的节点组进行检索
    **kwargs: 检索器配置参数，包括：
        - similarity (str): 相似度计算方法，'cosine'（余弦相似度）或'bm25'（BM25算法）
        - 其他检索器特定参数

**Returns:**\n
- self: 支持链式调用。
''')

add_chinese_doc('rag.retriever.WeightedRetriever', '''
WeightedRetriever 用于将多个 Retriever 的召回结果按照权重进行融合。

该组合器要求：
- **禁止使用 priority**：所有子 Retriever 不允许定义 priority 属性；
- **权重一致性**：一旦任意 Retriever 定义了 weight，则所有 Retriever 都必须定义 weight；
- **按比例分配 Top-K**：在设置 topk 的情况下，根据各 Retriever 的权重比例分配返回名额，
  并在部分 Retriever 结果不足时，动态将剩余额度重新分配给其他 Retriever；
- **支持权重归一化**：内部会自动对权重进行归一化处理，保证比例分配的稳定性。

适用于希望通过权重精细控制不同召回器贡献度的场景，例如：
BM25 + 向量检索 + 规则检索的加权融合。
''')

add_english_doc('rag.retriever.WeightedRetriever', '''
WeightedRetriever combines multiple Retrievers by weighting their retrieval results.

Key characteristics:
- **Priority is not allowed**: Sub-retrievers must not define a priority attribute.
- **Weight consistency enforced**: If any retriever defines a weight, all retrievers must define one.
- **Proportional Top-K allocation**: When topk is specified, results are allocated proportionally
  according to weights, with dynamic reallocation if some retrievers return fewer results than expected.
- **Automatic weight normalization**: Weights are normalized internally to ensure stable proportional behavior.

This retriever is suitable for scenarios where fine-grained control over the contribution of
different retrieval strategies (e.g., BM25, vector search, rule-based retrieval) is required.
''')

add_chinese_doc('rag.retriever.PriorityRetriever', '''
PriorityRetriever 用于基于优先级对多个 Retriever 的结果进行组合。

该组合器的设计原则包括：
- **禁止使用 weight**：子 Retriever 不允许定义 weight 属性；
- **基于优先级顺序返回结果**：按照 high → normal → low 的顺序依次合并各 Retriever 的结果；
- **支持 ignore 优先级**：被标记为 ignore 的 Retriever 将在预处理阶段被直接跳过；
- **Top-K 截断**：在合并过程中一旦达到 topk 数量即停止继续合并。

适用于对结果顺序要求明确、需要“高优先级结果优先返回”的场景，
例如规则召回优先于语义召回的检索体系。
''')

add_english_doc('rag.retriever.PriorityRetriever', '''
PriorityRetriever combines multiple Retrievers based on predefined priority levels.

Design principles:
- **Weights are not allowed**: Sub-retrievers must not define a weight attribute.
- **Priority-ordered merging**: Results are merged in the order of
  high → normal → low priority.
- **Ignore support**: Retrievers marked with the ignore priority are skipped during preprocessing.
- **Top-K cutoff**: Merging stops as soon as the accumulated result size reaches topk.

This retriever is suitable for scenarios where strict ordering is required and
high-priority retrieval results must be returned before others, such as
rule-based retrieval taking precedence over semantic retrieval.
''')

add_chinese_doc('rag.retriever.WeightedRetriever.forward', '''
执行加权检索并融合多个 Retriever 的召回结果。

该方法会：
- 根据传入或预定义的 weights 对 Retriever 进行加权；
- 自动过滤权重接近 0 的 Retriever，以减少不必要的召回开销；
- 在设置 topk 时，按权重比例对各 Retriever 的结果进行名额分配，
  并在结果不足时进行动态回填；
- 支持自定义 combine 函数以覆盖默认的融合逻辑。

Args:
    query (str): 用户查询文本。
    filters (dict, optional): 检索过滤条件。
    weights (List[float], optional): 每个 Retriever 对应的权重列表。
    topk (int, optional): 最终返回的最大结果数量。
    combine (Callable, optional): 自定义结果融合函数。
''')

add_english_doc('rag.retriever.WeightedRetriever.forward', '''
Execute weighted retrieval and combine results from multiple Retrievers.

This method:
- Applies provided or predefined weights to each Retriever;
- Filters out retrievers with near-zero weights to reduce unnecessary retrieval cost;
- Allocates Top-K slots proportionally based on weights, with dynamic redistribution
  when some retrievers return fewer results than expected;
- Allows a custom combine function to override the default merging logic.

Args:
    query (str): User query string.
    filters (dict, optional): Retrieval filter conditions.
    weights (List[float], optional): Weight list corresponding to each Retriever.
    topk (int, optional): Maximum number of results to return.
    combine (Callable, optional): Custom result combination function.
''')

add_chinese_doc('rag.retriever.PriorityRetriever.forward', '''
执行基于优先级的检索并合并多个 Retriever 的结果。

该方法会：
- 使用传入或 Retriever 自身定义的 priorities；
- 在预处理阶段直接忽略 priority 为 ignore 的 Retriever；
- 按 high → normal → low 的顺序合并各 Retriever 的结果；
- 在达到 topk 数量后立即停止合并，避免不必要的计算。

Args:
    query (str): 用户查询文本。
    filters (dict, optional): 检索过滤条件。
    priorities (List[Retriever.Priority], optional): 每个 Retriever 的优先级列表。
    topk (int, optional): 最终返回的最大结果数量。
    combinef (Callable, optional): 自定义优先级融合函数。
''')

add_english_doc('rag.retriever.PriorityRetriever.forward', '''
Execute priority-based retrieval and merge results from multiple Retrievers.

This method:
- Uses provided priorities or those defined on each Retriever;
- Skips retrievers with the ignore priority during preprocessing;
- Merges results in the order of high → normal → low priority;
- Stops merging as soon as the accumulated results reach topk.

Args:
    query (str): User query string.
    filters (dict, optional): Retrieval filter conditions.
    priorities (List[Retriever.Priority], optional): Priority list for each Retriever.
    topk (int, optional): Maximum number of results to return.
    combinef (Callable, optional): Custom priority-based combination function.
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

add_chinese_doc('rag.document.UrlDocument.get_nodes', '''\
按条件获取远程文档节点列表。

Args:
    uids (Optional[List[str]]): 指定节点 uid 列表。
    doc_ids (Optional[Set]): 指定文档 id 集合。
    group (Optional[str]): 节点组名。
    kb_id (Optional[str]): 知识库 id。
    numbers (Optional[Set]): 节点编号集合。

**Returns:**\n
- List[DocNode]: 命中的节点列表。
''')

add_english_doc('rag.document.UrlDocument.get_nodes', '''\
Get remote document nodes by criteria.

Args:
    uids (Optional[List[str]]): List of node uids to fetch.
    doc_ids (Optional[Set]): Set of document ids to filter by.
    group (Optional[str]): Node group name.
    kb_id (Optional[str]): Knowledge base id.
    numbers (Optional[Set]): Set of node numbers.

**Returns:**\n
- List[DocNode]: Matched nodes.
''')

add_chinese_doc('rag.document.UrlDocument.get_window_nodes', '''\
获取远程文档中指定节点的窗口节点。

Args:
    node (DocNode): 目标节点。
    span (tuple[int, int]): 窗口范围，基于 node.number 的相对偏移。
    merge (bool): 是否将窗口节点合并为一个节点返回。

**Returns:**\n
- Union[List[DocNode], DocNode]: 窗口节点列表，或合并后的单节点。
''')

add_english_doc('rag.document.UrlDocument.get_window_nodes', '''\
Get window nodes around a target node in a remote document.

Args:
    node (DocNode): Target node.
    span (tuple[int, int]): Window range based on relative offsets of node.number.
    merge (bool): Whether to merge window nodes into a single node.

**Returns:**\n
- Union[List[DocNode], DocNode]: Window nodes list or a merged node.
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

add_english_doc('rag.doc_node.DocNodeCore', '''
Core document node class for text and metadata storage.

Args:
    text (Optional[str]): Node text content.
    metadata (Optional[Dict[str, Any]]): Node-level metadata.
    uid (Optional[str]): Unique node identifier. If not provided, a UUID is generated.
''')

add_chinese_doc('rag.doc_node.DocNodeCore', '''
文档节点核心类，用于保存文本与元数据。

Args:
    text (Optional[str]): 节点文本内容。
    metadata (Optional[Dict[str, Any]]): 节点级元数据。
    uid (Optional[str]): 节点唯一标识；未提供时自动生成 UUID。
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

add_chinese_doc('rag.doc_node.DocNode.set_embedding', """\
设置文档节点的嵌入向量。

为文档节点设置指定键的嵌入向量值，用于后续的检索和相似度计算。

Args:
    embed_key (str): 嵌入向量的键名
    embed_value: 嵌入向量的值

Returns:
    None
""")

add_english_doc('rag.doc_node.DocNode.set_embedding', """\
Set embedding vector for document node.

Set the embedding vector value for specified key in document node, used for subsequent retrieval and similarity calculation.

Args:
    embed_key (str): Key name of the embedding vector
    embed_value: Value of the embedding vector

Returns:
    None
""")

add_chinese_doc('rag.doc_node.DocNode.copy', """
复制当前 DocNode，并生成新的 uid。

复制后的节点会记录来源信息（_copy_source），可选更新 metadata/global_metadata。

Args:
    global_metadata (dict): 需要合并到 global_metadata 的字段
    metadata (dict): 需要合并到 metadata 的字段

Returns:
    DocNode: 复制后的节点
""")

add_english_doc('rag.doc_node.DocNode.copy', """
Copy the current DocNode and generate a new uid.

The copied node records its source (_copy_source) and can optionally merge metadata/global_metadata.

Args:
    global_metadata (dict): Fields to merge into global_metadata
    metadata (dict): Fields to merge into metadata

Returns:
    DocNode: The copied node
""")

add_chinese_doc('rag.parsing_service.server.DocumentProcessor', """
文档处理服务类，启动后可对外提供文档处理服务，支持文档的添加、删除和更新等操作。
服务内部采取生产者-消费者模式，通过队列管理文档处理任务，支持异步处理文档任务，支持任务状态回调通知。

Args:
    port (Optional[int]): 服务端口号。默认为None，当为None时，将自动分配端口。
    url (Optional[str]): 服务URL，提供服务URL时，模块可远程连接已经部署好的服务，无需再启动服务，默认为None。
    num_workers (int): 工作线程数，默认为1，当为0时，不启动工作线程，仅启动服务。
    db_config (Optional[Dict[str, Any]]): 用于配置SqlManager实现数据库连接，默认为None，当为None时，使用默认数据库配置。
    launcher (Optional[Launcher]): 用于管理服务进程的Launcher实例，默认为None。
    post_func (Optional[Callable]): 用于处理任务状态回调通知的函数，默认为None，当为None时，不进行任务状态回调通知,必须提供一个函数，函数签名如下：
        def post_func(task_id: str, task_status: str = None, error_code: str = None, error_msg: str = None):
            pass
    path_prefix (Optional[str]): 用于配置上传文件存储路径前缀，默认为None。
    lease_duration (float): 任务租约时长（秒），默认为300。
    lease_renew_interval (float): 租约续租间隔（秒），默认为60。
    high_priority_task_types (Optional[List[str]]): 高优任务类型列表，默认包含 DOC_DELETE。
    high_priority_workers (int): 高优任务 worker 数量，默认1。
""")

add_english_doc('rag.parsing_service.server.DocumentProcessor', """
Document processing service class, after startup, it can provide document processing services, supporting document addition, deletion and update operations.
The service internally adopts a producer-consumer model, manages document processing tasks through a queue, supports asynchronous processing of document tasks, and supports task status callback notifications.

Args:
    port (Optional[int]): Service port number. Defaults to None, when it is None, a random port will be assigned.
    url (Optional[str]): Service URL, when the service URL is provided, the module can remotely connect to the already deployed service, without starting the service again, defaults to None.
    num_workers (int): Number of worker threads, defaults to 1, when it is 0, the worker threads are not started, only the service is started.
    db_config (Optional[Dict[str, Any]]): Used to configure the database connection information for SqlManager, defaults to None, when it is None, the default database configuration is used.
    launcher (Optional[Launcher]): Used to manage the Launcher instance of the service process, defaults to None.
    post_func (Optional[Callable]): Used to process the task status callback notification function, defaults to None, when it is None, the task status callback notification is not performed, must provide a function, the function signature is as follows:
        def post_func(task_id: str, task_status: str = None, error_code: str = None, error_msg: str = None):
            pass
    path_prefix (Optional[str]): Used to configure the prefix of the uploaded file storage path, defaults to None.
    lease_duration (float): Task lease duration in seconds, defaults to 300.
    lease_renew_interval (float): Lease renewal interval in seconds, defaults to 60.
    high_priority_task_types (Optional[List[str]]): High priority task types, defaults to [DOC_DELETE].
    high_priority_workers (int): Number of high priority workers, defaults to 1.
""")

add_example('rag.parsing_service.server.DocumentProcessor', """
```python
# set db_config
db_config = {
    'db_type': 'sqlite',
    'user': None,
    'password': None,
    'host': None,
    'port': None,
    'db_name': '/xxx/xxx/test.db',
}
# Create server and start it
server = DocumentProcessor(port=28888, db_config=db_config, num_workers=4, post_func=post_func_sample)
server.start()

# start the document with server
server = DocumentProcessor(port=28888, db_config=db_config, num_workers=4, post_func=post_func_sample)
document = Document(dataset_path=None, name="algo_1", display_name="Algo_1",
                    description="Algo_1 for testing", manager=server)
document.start()

# Create remote document processor
remote_server = DocumentProcessor(url="http://remote-server:8080")
document = Document(dataset_path=None, name="algo_1", display_name="Algo_1",
                    description="Algo_1 for testing", manager=remote_server)
document.start()
```
""")

add_chinese_doc('rag.parsing_service.server.DocumentProcessor.register_algorithm', """
注册算法到文档处理服务，内部会自动将算法信息存储到数据库中，后续可使用该算法处理文档。
该方法必须与 Document 模块配合使用，才能正常工作，一般无需自行调用。

Args:
    name (str): 算法名称，作为唯一标识符。
    store (_DocumentStore): _DocumentStore实例，用于管理文档数据。
    reader (DirectoryReader): 读取器实例，用于解析文档内容。
    node_groups (Dict[str, Dict]): 节点组配置信息。
    display_name (Optional[str]): 算法的显示名称，默认为None。
    description (Optional[str]): 算法的描述信息，默认为None。
""")

add_english_doc('rag.parsing_service.server.DocumentProcessor.register_algorithm', """
Register an algorithm to the document processing service.
The algorithm information will be automatically stored in the database, and can be used to process documents later.
This method must be used with the Document module to work properly, and generally does not need to be called manually.

Args:
    name (str): Algorithm name as unique identifier.
    store (_DocumentStore): _DocumentStore instance for managing document data.
    reader (DirectoryReader): Reader instance for parsing document content.
    node_groups (Dict[str, Dict]): Node group configuration information.
    display_name (Optional[str]): Display name for the algorithm, defaults to None.
    description (Optional[str]): Description of the algorithm, defaults to None.
""")

add_chinese_doc('rag.parsing_service.server.DocumentProcessor.drop_algorithm', """
从文档处理服务中移除指定算法， 该方法会自动从数据库中删除算法信息，后续无法使用该算法处理文档。

Args:
    name (str): 要移除的算法唯一标识。
""")

add_english_doc('rag.parsing_service.server.DocumentProcessor.drop_algorithm', """
Remove specified algorithm from document processing service. This method will automatically delete the algorithm information from the database, and the algorithm will no longer be available for subsequent use.

Args:
    name (str): Unique identifier of the algorithm to remove.
""")

add_chinese_doc('rag.parsing_service.server.DocumentProcessor.start', '''
启动文档处理服务，该方法会启动服务端口，并启动工作线程，后续可使用该服务处理文档。若初始化时设置了工作线程数大于0，则会启动工作线程，否则仅启动服务。
''')

add_english_doc('rag.parsing_service.server.DocumentProcessor.start', '''
Start the document processing service.
This method will start the service port and start the worker threads, and subsequent documents can be processed using this service.
If the worker thread number is set to greater than 0 in the service, the worker threads will be started, otherwise only the service will be started.
''')

add_chinese_doc('rag.parsing_service.worker.DocumentProcessorWorker', '''
文档处理消费者线程类，启动后将负责处理文档处理服务中的任务，并将其结果返回给服务。
模块支持独立部署，也可直接在 DocumentProcessor 中通过设置 num_workers 参数自动启动工作线程。

Args:
    db_config (Optional[Dict[str, Any]]): 用于配置SqlManager实现数据库连接，默认为None，当为None时，使用默认数据库配置。
    num_workers (int): 工作线程数，默认为1， 当大于1时，内部基于ray集群启动多个工作线程，否则仅启动一个工作线程。
    port (Optional[int]): 服务端口号。默认为None，当为None时，将自动分配端口。
    task_poller (Optional[Callable]): 外部任务拉取函数，可选。
    poll_mode (str): 任务拉取模式，可选值为 "direct" 或 "thread"。
        - "direct": 不启动独立 poller 线程，worker 空闲时直接拉取并立即处理任务（默认）。
        - "thread": 启动独立 poller 线程，持续拉取任务并入队。
    lease_duration (float): 任务租约时长（秒），默认为300。
    lease_renew_interval (float): 租约续租间隔（秒），默认为60。
    high_priority_task_types (Optional[List[str]]): 高优任务类型列表，可选。
    high_priority_only (bool): 仅处理高优任务，默认为False。
''')

add_english_doc('rag.parsing_service.worker.DocumentProcessorWorker', '''
Document processing consumer thread class, after startup, it will be responsible for processing tasks in the document processing service, and returning the results to the service.
The module supports independent deployment, or can automatically start worker threads by setting the num_workers parameter in DocumentProcessor.

Args:
    db_config (Optional[Dict[str, Any]]): Used to configure the database connection information for SqlManager, defaults to None, when it is None, the default database configuration is used.
    num_workers (int): Number of worker threads, defaults to 1, when it is greater than 1, multiple worker threads are started internally based on the ray cluster, otherwise only one worker thread is started.
    port (Optional[int]): Service port number. Defaults to None, when it is None, a random port will be assigned.
    task_poller (Optional[Callable]): External task poller callback, optional.
    poll_mode (str): Task polling mode, either "direct" or "thread".
        - "direct": No dedicated poller thread; the worker pulls and processes tasks when idle (default).
        - "thread": Run a dedicated poller thread to continuously fetch tasks and enqueue them.
    lease_duration (float): Task lease duration in seconds, defaults to 300.
    lease_renew_interval (float): Lease renewal interval in seconds, defaults to 60.
    high_priority_task_types (Optional[List[str]]): High priority task types, optional.
    high_priority_only (bool): Process high priority tasks only, defaults to False.
''')

add_chinese_doc('rag.parsing_service.worker.DocumentProcessorWorker.start', '''
启动文档处理消费者线程，该方法会启动工作线程，并启动服务端口，后续可使用该服务处理文档。若初始化时设置了工作线程数大于1，则会启动多个工作线程，否则仅启动一个工作线程。
''')

add_english_doc('rag.parsing_service.worker.DocumentProcessorWorker.start', '''
Start the document processing consumer thread. This method will start the worker threads and start the service port, and subsequent documents can be processed using this service.
If the worker thread number is set to greater than 1 in the initialization, multiple worker threads will be started, otherwise only one worker thread will be started.
''')

add_example('rag.parsing_service.worker.DocumentProcessorWorker', '''
```python
db_config = {
    'db_type': 'sqlite',
    'user': None,
    'password': None,
    'host': None,
    'port': None,
    'db_name': '/xxx/xxx/test.db',
}
# Create worker and start it
worker = DocumentProcessorWorker(db_config=db_config, num_workers=2, port=28888)
worker.start()
```
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader', '''
A modular document directory reader that inherits from ModuleBase, supporting reading various document formats from the file system and converting them into standardized DocNode objects.

This class supports direct file input or directory input (mutually exclusive). It provides built-in readers for common formats such as PDF, DOCX, PPTX, images, CSV, Excel, audio/video, etc., while also allowing users to register custom file readers.

Args:
    input_dir (Optional[str]): Input directory path. Mutually exclusive with input_files.
                               Must exist in the file system if provided.
    input_files (Optional[List]): Directly specified list of files. Mutually exclusive with input_dir.
                                  Each file must exist either in the provided path or under `config['data_path']`.
    exclude (Optional[List]): List of file patterns to exclude from processing.
    exclude_hidden (bool): Whether to exclude hidden files. Defaults to True.
    recursive (bool): Whether to recursively read subdirectories. Defaults to False.
    encoding (str): Encoding format of text files. Defaults to "utf-8".
    filename_as_id (bool): Deprecated argument. No longer used. A warning will be logged if provided.
    required_exts (Optional[List[str]]): Whitelist of file extensions to process. Only files with these extensions will be read.
    file_extractor (Optional[Dict[str, Callable]]): Dictionary of custom file readers. Keys are filename patterns, values are reader callables.
    fs (Optional[AbstractFileSystem]): Custom file system to use. Defaults to the system's default file system.
    metadata_genf (Optional[Callable[[str], Dict]]): Metadata generation function that takes a file path and returns a metadata dictionary.
                                                     Defaults to an internal implementation (_DefaultFileMetadataFunc).
    num_files_limit (Optional[int]): Maximum number of files to read. If exceeded, only the first N files are processed.
    return_trace (bool): Whether to return processing trace information. Defaults to False.
    metadatas (Optional[Dict]): Predefined global metadata dictionary to attach to all documents.
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader', '''
模块化的文档目录读取器，继承自 ModuleBase，支持从文件系统读取多种格式的文档并转换为标准化的 DocNode 。

该类支持直接指定文件列表或输入目录（二者互斥）。内置了对常见格式（如 PDF、DOCX、PPTX、图片、CSV、Excel、音视频等）的支持，也允许用户注册自定义的文件读取器。

Args:
    input_dir (Optional[str]): 输入目录路径。与 input_files 互斥。目录必须存在。
    input_files (Optional[List]): 直接指定的文件列表。与 input_dir 互斥。文件必须存在于指定路径或 `config['data_path']` 下。
    exclude (Optional[List]): 需要排除的文件模式列表。
    exclude_hidden (bool): 是否排除隐藏文件。默认为 True。
    recursive (bool): 是否递归读取子目录。默认为 False。
    encoding (str): 文本文件的编码格式。默认为 "utf-8"。
    filename_as_id (bool): 已弃用参数，不再使用。如果提供会打印警告日志。
    required_exts (Optional[List[str]]): 需要处理的文件扩展名白名单。仅处理这些扩展名的文件。
    file_extractor (Optional[Dict[str, Callable]]): 自定义文件读取器字典。键为文件名模式，值为读取器函数。
    fs (Optional[AbstractFileSystem]): 自定义文件系统。默认为系统的默认文件系统。
    metadata_genf (Optional[Callable[[str], Dict]]): 元数据生成函数，接收文件路径返回元数据字典。默认为内部实现 (_DefaultFileMetadataFunc)。
    num_files_limit (Optional[int]): 最大读取文件数量限制。超过时仅处理前 N 个文件。
    return_trace (bool): 是否返回处理过程追踪信息。默认为 False。
    metadatas (Optional[Dict]): 预定义的全局元数据字典，将附加到所有文档上。
''')

add_example('rag.dataReader.SimpleDirectoryReader', '''
>>> import lazyllm
>>> from lazyllm.tools.dataReader import SimpleDirectoryReader
>>> reader = SimpleDirectoryReader(input_dir="yourpath/",recursive=True,exclude=["*.tmp"],required_exts=[".pdf", ".docx"])
>>> documents = reader.load_data()
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader.load_file', '''\
使用指定的 Reader 将单个文件加载为 `DocNode` 列表。

该方法会根据文件名模式匹配合适的读取器（reader），并遵循以下优先级生成元数据：
`用户提供 > reader 自动生成 > metadata_genf 生成`。
在配置允许的情况下支持回退到原始文本读取。

Args:
    input_file (Path): 要读取的文件路径。
    metadata_genf (Callable): 根据文件路径生成元数据的函数。
    file_extractor (Dict[str, Callable]): 文件扩展名模式与 reader 的映射表。
    encoding (str): 文件读取时使用的文本编码，默认为 "utf-8"。
    pathm (PurePath): 路径处理模块，支持本地或远程路径。
    fs (AbstractFileSystem): 可选文件系统对象，兼容 fsspec 抽象。
    metadata (Dict): 可选用户自定义元数据，优先于自动生成。

**Returns:**\n
- List[DocNode]: 从文件中提取的文档对象列表。
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader.load_file', '''\
Load a single file into a list of `DocNode` objects using the appropriate reader.

This method selects the appropriate reader based on filename patterns and applies metadata with the following priority:
`user > reader > metadata_genf`.
Optionally falls back to raw text decoding depending on config.

Args:
    input_file (Path): Path to the input file.
    metadata_genf (Callable): Function to generate metadata from file path.
    file_extractor (Dict[str, Callable]): Mapping of filename patterns to reader callables.
    encoding (str): Text encoding to use when reading files. Default is "utf-8".
    pathm (PurePath): Path handling module for local or remote paths.
    fs (AbstractFileSystem): Optional filesystem abstraction from fsspec.
    metadata (Dict): Optional user-defined metadata overriding auto-generated ones.

**Returns:**\n
- List[DocNode]: List of parsed documents extracted from the file.
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader.find_extractor_by_file', '''
根据文件名或后缀从文件读取器映射中选择合适的提取器（extractor）。

该函数首先尝试使用文件后缀进行直接匹配（如 `*.txt`），
若未命中，则会遍历 `file_extractor` 的模式键（如 `*.json`, `**/docs/*.md`），
使用 `fnmatch` 进行模糊匹配，找到最符合的读取器。
如果没有匹配项，将返回默认读取器 `DefaultReader`。

Args:
    input_file (Path): 输入文件路径。
    file_extractor (Dict[str, Callable]): 文件模式到提取器的映射表。
    pathm (PurePath): 路径处理模块，用于生成匹配模式，默认使用 `Path`。

**Returns:**\n
- Callable: 与文件匹配的提取器函数，若无匹配则返回 `DefaultReader`。
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader.find_extractor_by_file', '''
Select the appropriate file extractor based on filename or suffix.

This function first attempts to match by file extension (e.g., `*.txt`),
and if no match is found, it iterates through the `file_extractor` mapping,
using `fnmatch` for wildcard-based pattern matching (e.g., `*.json`, `**/docs/*.md`).
If no extractor matches, it falls back to the `DefaultReader`.

Args:
    input_file (Path): Path to the input file.
    file_extractor (Dict[str, Callable]): Mapping of filename patterns to extractor functions.
    pathm (PurePath): Path handling module used to construct pattern paths. Defaults to `Path`.

**Returns:**\n
- Callable: The extractor function matching the file, or `DefaultReader` if none found.
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader.get_default_reader', '''
根据文件扩展名获取默认的文件读取器（Reader）。

该函数通过文件扩展名（如 `.txt`、`.json`）在默认读取器映射表中查找对应的 Reader，
若未以 `"*."` 开头，会自动补全后缀格式（例如 `"txt"` → `"*.txt"`）。
常见的默认 Reader 包括纯文本读取器、JSON 读取器、Markdown 读取器等。

Args:
    file_ext (str): 文件扩展名或匹配模式（例如 `"txt"` 或 `"*.json"`）。

**Returns:**\n
- Callable[[Path, Dict], List[DocNode]]: 与该扩展名对应的读取器函数，若未匹配则返回 `None`。
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader.get_default_reader', '''
Retrieve the default file reader (Reader) based on file extension.

This function looks up the default reader mapping using the file extension
(e.g., `.txt`, `.json`).
If the extension does not start with `"*."`, it automatically prepends it
(e.g., `"txt"` → `"*.txt"`).
Common readers include plain text, JSON, and Markdown readers.

Args:
    file_ext (str): File extension or matching pattern (e.g., `"txt"` or `"*.json"`).

**Returns:**\n
- Callable[[Path, Dict], List[DocNode]]: The reader function associated with the extension, or `None` if not found.
''')

add_chinese_doc('rag.dataReader.SimpleDirectoryReader.add_post_action_for_default_reader', '''
为默认 Reader 添加后处理函数（Post Action）。

该方法允许在默认文件读取器（Reader）完成文档解析后，对生成的 `DocNode`
进行自定义后处理（如文本清洗、节点拆分、结构调整等）。
若指定的扩展名没有默认读取器，会抛出 `KeyError` 异常。

后处理函数可以是以下类型之一：

1. 继承自 `NodeTransform` 的类；
2. 普通函数，接收一个 `DocNode` 并返回修改后的 `DocNode` 或列表；
3. 可实例化的类型，会自动创建实例。

Args:
    file_ext (str): 文件扩展名或匹配模式（例如 `"*.txt"`）。
    f (Callable[[DocNode], Union[DocNode, List[DocNode]]]): 后处理函数或节点转换类。

**Raises:**\n
- KeyError: 当指定文件扩展名没有默认 Reader 时抛出。
''')

add_english_doc('rag.dataReader.SimpleDirectoryReader.add_post_action_for_default_reader', '''
Add a post-processing action (Post Action) for a default Reader.

This method allows attaching a custom post-processing function to the default
file reader (Reader), enabling transformation of parsed `DocNode` objects after
initial loading (e.g., text cleaning, node splitting, or structural adjustments).
If the given file extension has no default reader, a `KeyError` is raised.

The post-processing function `f` can be:

1. A subclass of `NodeTransform`;
2. A callable that takes a `DocNode` and returns a modified `DocNode` or a list;
3. A class type, which will be instantiated automatically.

Args:
    file_ext (str): File extension or matching pattern (e.g., `"*.txt"`).
    f (Callable[[DocNode], Union[DocNode, List[DocNode]]]): Post-processing function or node transform class.

**Raises:**\n
- KeyError: If the specified file extension has no default reader.
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
基础文档读取器类，提供文档加载的基本接口。继承自 ModuleBase，使用 LazyLLMRegisterMetaClass 作为元类。

Args:
    *args: 位置参数，保留给子类或父类使用。
    return_trace (bool): 是否返回处理过程的追踪信息，默认为 True。
    **kwargs: 关键字参数，保留给子类或父类使用。
''')

add_english_doc('rag.readers.readerBase.LazyLLMReaderBase', '''
Base document reader class that provides fundamental interfaces for document loading. Inherits from ModuleBase and uses LazyLLMRegisterMetaClass as metaclass.

Args:
    *args: Positional arguments, reserved for parent or subclass use.
    return_trace (bool): Whether to return processing trace information. Defaults to True.
    **kwargs: Keyword arguments, reserved for parent or subclass use.
''')

add_example('rag.readers.readerBase.LazyLLMReaderBase', '''
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
''')

add_english_doc('rag.readers.readerBase.LazyLLMReaderBase.detect_encoding', '''\
Detect the encoding of a file.

Args:
    file_path (str): The path of the file.
    fs (fsspec.AbstractFileSystem): The file system.
    sample_size (int): The sample size.
    use_cache (bool): Whether to use cache.
    enable_chardet (bool): Whether to enable chardet.

**Returns:**\n
- str: The encoding of the file.
''')

add_chinese_doc('rag.readers.readerBase.LazyLLMReaderBase.detect_encoding', '''\
检测文件的编码。

Args:
    file_path (str): 文件路径。
    fs (fsspec.AbstractFileSystem): 文件系统。
    sample_size (int): 样本大小。
    use_cache (bool): 是否使用缓存。
    enable_chardet (bool): 是否启用 chardet。

**Returns:**\n
- str: 文件的编码。
''')

add_example('rag.readers.readerBase.LazyLLMReaderBase.detect_encoding', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.readers import LazyLLMReaderBase
>>> reader = LazyLLMReaderBase()
>>> encoding = reader.detect_encoding("path/to/file.txt")
>>> print(encoding)
''')

add_english_doc('rag.readers.readerBase.LazyLLMReaderBase.clear_encoding_cache', '''\
Clear the encoding cache.

Args:
    file_path (str): The path of the file.
    fs (fsspec.AbstractFileSystem): The file system.
    sample_size (int): The sample size.
    use_cache (bool): Whether to use cache.
    enable_chardet (bool): Whether to enable chardet.
''')

add_chinese_doc('rag.readers.readerBase.LazyLLMReaderBase.clear_encoding_cache', '''\
清空编码缓存。

Args:
    file_path (str): 文件路径。
    fs (fsspec.AbstractFileSystem): 文件系统。
    sample_size (int): 样本大小。
    use_cache (bool): 是否使用缓存。
    enable_chardet (bool): 是否启用 chardet。
''')

add_example('rag.readers.readerBase.LazyLLMReaderBase.clear_encoding_cache', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.readers import LazyLLMReaderBase
>>> reader = LazyLLMReaderBase()
>>> reader.clear_encoding_cache()
''')

add_english_doc('rag.readers.readerBase.LazyLLMReaderBase.get_encoding_cache_stats', '''\
Get the encoding cache stats.

**Returns:**\n
- dict: The encoding cache stats.
''')

add_chinese_doc('rag.readers.readerBase.LazyLLMReaderBase.get_encoding_cache_stats', '''\
获取编码缓存统计信息。

**Returns:**\n
- dict: 编码缓存统计信息。
''')

add_example('rag.readers.readerBase.LazyLLMReaderBase.get_encoding_cache_stats', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag.readers import LazyLLMReaderBase
>>> reader = LazyLLMReaderBase()
>>> stats = reader.get_encoding_cache_stats()
>>> print(stats)
''')

add_example('rag.readers.MineruPDFReader', '''\
from lazyllm.tools.rag.readers import MineruPDFReader
reader = MineruPDFReader("http://0.0.0.0:8888")  # Mineru server address
nodes = reader("path/to/pdf")
''')
add_chinese_doc('rag.readers.readerBase.TxtReader', '''\
TxtReader 类用于从文本文件中加载内容，并将其封装为 `DocNode` 对象列表。

该类继承自 `LazyLLMReaderBase`，主要功能包括：

- 支持指定文本编码读取文件；
- 可选返回加载过程的跟踪信息；

Args:
    encoding (str): 文件读取的文本编码，默认值为 'utf-8'。
    return_trace (bool): 是否返回加载过程的跟踪信息，默认值为 True。
''')

add_english_doc('rag.readers.readerBase.TxtReader', '''\
The TxtReader class loads content from text files and wraps it into a list of `DocNode` objects.

This class inherits from `LazyLLMReaderBase` and mainly provides:

- Support for reading files with a specified text encoding;
- Optional tracing information of the loading process;

Args:
    encoding (str): Text encoding for reading files, default is 'utf-8'.
    return_trace (bool): Whether to return trace information of the loading process, default is True.
''')

add_chinese_doc('rag.doc_node.QADocNode', '''\
问答文档节点类，用于存储问答对数据。

Args:
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

Args:
    metadata_mode (MetadataMode): 元数据模式，默认为MetadataMode.NONE。
        当设置为MetadataMode.LLM时，返回格式化的问答对。
        其他模式下返回基类的文本格式。

**Returns:**\n
- str: 格式化后的文本内容。
''')

add_english_doc('rag.doc_node.QADocNode.get_text', '''\
Get the text content of the node.

Args:
    metadata_mode (MetadataMode): Metadata mode, defaults to MetadataMode.NONE.
        When set to MetadataMode.LLM, returns formatted QA pair.
        For other modes, returns base class text format.

**Returns:**\n
- str: The formatted text content.
''')

# ---------------------------------------------------------------------------- #

# rag/transform

add_english_doc('rag.transform.RichTransform', '''
Transform a `RichDocNode` into a list of `DocNode` objects, and preserve the metadata of each `DocNode`.
The input must be a `RichDocNode` instance.

Args:
    node (RichDocNode): Rich document node to unwrap.

Returns:
    List[DocNode]: The underlying node list.
''')

add_chinese_doc('rag.transform.RichTransform', '''
将 `RichDocNode` 拆分为 `DocNode` 列表，并保留每个 `DocNode` 的元数据。
输入必须是 `RichDocNode` 实例。

Args:
    node (RichDocNode): 需要拆分的富文档节点。

Returns:
    List[DocNode]: 拆分后的节点列表。
''')

add_example('rag.transform.RichTransform', '''
>>> from lazyllm.tools.rag.transform import RichTransform
>>> nodes = RichTransform().transform(rich_node)
''')

add_english_doc('rag.transform.sentence.SentenceSplitter', '''
Split sentences into chunks of a specified size. You can specify the size of the overlap between adjacent chunks.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    **kwargs: Additional parameters passed to the splitter.
''')

add_chinese_doc('rag.transform.sentence.SentenceSplitter', '''
将句子拆分成指定大小的块。可以指定相邻块之间重合部分的大小。

Args:
    chunk_size (int): 拆分之后的块大小
    chunk_overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量
    **kwargs: 传递给拆分器的额外参数。
''')

add_example('rag.transform.sentence.SentenceSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, SentenceSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
''')
add_chinese_doc('rag.transform.sentence.SentenceSplitter.set_default', '''
设置SentenceSplitter全局的默认参数。

Args:
    **kwargs: parameters passed to the splitter.
''')

add_english_doc('rag.transform.character.CharacterSplitter', '''
Split text by characters.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    separator (str): The separator to use for splitting. Defaults to ' '.
    is_separator_regex (bool): Whether the separator is a regular expression. Defaults to False.
    keep_separator (bool): Whether to keep the separator in the split text. Defaults to False.
    **kwargs: Additional parameters passed to the splitter.
''')

add_chinese_doc('rag.transform.character.CharacterSplitter', '''
将文本按字符拆分。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int): 控制并行处理的线程/进程数量。
    separator (str): 用于拆分的分隔符。默认为' '。
    is_separator_regex (bool): 是否使用正则表达式作为分隔符。默认为False。
    keep_separator (bool): 是否保留分隔符在拆分后的文本中。默认为False。
    **kwargs: 传递给拆分器的额外参数。
''')

add_example('rag.transform.character.CharacterSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import Document, CharacterSplitter
>>> m = lazyllm.OnlineEmbeddingModule(source="glm")
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="characters", transform=CharacterSplitter, chunk_size=1024, chunk_overlap=100)
''')

add_english_doc('rag.transform.character.CharacterSplitter.set_split_fns', '''
CharacterSplitter has default split functions, you can also set the split functions for the CharacterSplitter.
You can set multiple split functions, and the CharacterSplitter will use them in order, the separator parameter will be ignored.

Args:
    split_fns (List[Callable[[str], List[str]]]): The split functions to use.
''')

add_chinese_doc('rag.transform.character.CharacterSplitter.set_split_fns', '''
CharacterSplitter有默认的拆分函数，你也可以设置自己的拆分函数。
可以设置多个拆分函数，CharacterSplitter会按顺序使用这些函数，分隔符参数将失效。

Args:
    split_fns (List[Callable[[str], List[str]]]): 要使用的拆分函数列表。
''')

add_example('rag.transform.character.CharacterSplitter.set_split_fns', '''
>>> import lazyllm
>>> from lazyllm.tools import CharacterSplitter
>>> splitter = CharacterSplitter(separator='\n')
>>> splitter.set_split_fns([lambda text: text.split(' '), lambda text: text.split('\n')])
>>> text = 'Hello, world!'
>>> splits = splitter.split_text(text, metadata_size=0)
>>> print(splits)
''')

add_english_doc('rag.transform.character.CharacterSplitter.add_split_fn', '''
Add a split function to the CharacterSplitter.

Args:
    split_fn (Callable[[str], List[str]]): The split function to add.
    index (Optional[int]): The index to add the split function. Default to the last position.
    bind_separator (bool): Whether to bind the separator to the split function. Default to False.
''')

add_chinese_doc('rag.transform.character.CharacterSplitter.add_split_fn', '''
添加一个拆分函数到CharacterSplitter。

Args:
    split_fn (Callable[[str], List[str]]): 要添加的拆分函数。
    index (Optional[int]): 要添加的拆分函数的位置。默认为最后一个位置。
    bind_separator (bool): 是否将分隔符绑定到拆分函数。默认为False。
''')

add_example('rag.transform.character.CharacterSplitter.add_split_fn', '''
>>> import lazyllm
>>> from lazyllm.tools import CharacterSplitter
>>> splitter = CharacterSplitter(separator='\n')
>>> splitter.add_split_fn(lambda text: text.split(' '), index=0)
>>> text = 'Hello, world!'
>>> splits = splitter.split_text(text, metadata_size=0)
>>> print(splits)
''')

add_english_doc('rag.transform.character.CharacterSplitter.clear_split_fns', '''
Clear all split functions from the CharacterSplitter, and use the default split functions.
'''
)

add_chinese_doc('rag.transform.character.CharacterSplitter.clear_split_fns', '''
清除CharacterSplitter的所有拆分函数，并使用默认的拆分函数。
''')

add_example('rag.transform.character.CharacterSplitter.clear_split_fns', '''
>>> import lazyllm
>>> from lazyllm.tools import CharacterSplitter
>>> splitter = CharacterSplitter(separator='\n')
>>> splitter.clear_split_fns()
>>> text = 'Hello, world!'
>>> splits = splitter.split_text(text, metadata_size=0)
>>> print(splits)
''')

add_english_doc('rag.transform.recursive.RecursiveSplitter', '''
Split text by characters recursively.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    keep_separator (bool): Whether to keep the separator in the split text. Defaults to False.
    is_separator_regex (bool): Whether the separator is a regular expression. Defaults to False.
    separators (List[str]): The separators to use for splitting. Defaults to ['\n\n', '\n', ' ', '']. If you want to split by multiple separators, you can set this parameter.
''')

add_chinese_doc('rag.transform.recursive.RecursiveSplitter', '''
递归拆分文本。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量。
    keep_separator (bool): 是否保留分隔符在拆分后的文本中。默认为False。
    is_separator_regex (bool): 是否使用正则表达式作为分隔符。默认为False。
    separators (List[str]): 用于拆分的分隔符列表。默认为['\n\n', '\n', ' ', '']。如果你想按多个分隔符拆分，可以设置这个参数。
''')

add_example('rag.transform.recursive.RecursiveSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import RecursiveSplitter
>>> splitter = RecursiveSplitter(separators=['\n\n', '\n', ' ', ''])
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="recursive", transform=RecursiveSplitter, chunk_size=1024, chunk_overlap=100)
''')

add_english_doc('rag.transform.markdown.MarkdownSplitter', '''
Split markdown text by headers recursively.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    keep_trace (bool): Whether to keep the trace of the markdown text. Defaults to False.
    keep_headers (bool): Whether to keep the headers in the split text. Defaults to False.
    keep_lists (bool): Whether to keep the lists in the split text. Defaults to False.
    keep_code_blocks (bool): Whether to keep the code blocks in the split text. Defaults to False.
    keep_tables (bool): Whether to keep the tables in the split text. Defaults to False.
    keep_images (bool): Whether to keep the images in the split text. Defaults to False.
    keep_links (bool): Whether to keep the links in the split text. Defaults to False.
    **kwargs: Additional parameters passed to the splitter.
''')

add_chinese_doc('rag.transform.markdown.MarkdownSplitter', '''
递归拆分markdown文本。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int): 控制并行处理的线程/进程数量。
    keep_trace (bool): 是否保留markdown文本的追踪。默认为False。
    keep_headers (bool): 是否保留headers在拆分后的文本中。默认为False。
    keep_lists (bool): 是否保留lists在拆分后的文本中。默认为False。
    keep_code_blocks (bool): 是否保留code blocks在拆分后的文本中。默认为False。
    keep_tables (bool): 是否保留tables在拆分后的文本中。默认为False。
    keep_images (bool): 是否保留images在拆分后的文本中。默认为False。
    keep_links (bool): 是否保留links在拆分后的文本中。默认为False。
    **kwargs: 传递给拆分器的额外参数。
''')

add_example('rag.transform.markdown.MarkdownSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import MarkdownSplitter
>>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
>>> documents.create_node_group(name="markdown", transform=MarkdownSplitter,
                                chunk_size=1024, chunk_overlap=100, keep_trace=True, keep_headers=True)
''')

add_english_doc('rag.transform.markdown.MarkdownSplitter.split_markdown_by_semantics', '''
Split markdown text by semantics.

Args:
    md_text (str): The markdown text to split.
    **kwargs: Additional parameters passed to the splitter.

**Returns:**\n
- List[_MdSplit]: The split text with markdown semantics in metadata.
''')

add_chinese_doc('rag.transform.markdown.MarkdownSplitter.split_markdown_by_semantics', '''
拆分markdown文本的语义。

Args:
    md_text (str): 要拆分的markdown文本。
    **kwargs: 传递给拆分器的额外参数。

**Returns:**\n
- List[_MdSplit]: 拆分后的文本，包含markdown语义的元数据。
''')

add_example('rag.transform.markdown.MarkdownSplitter.split_markdown_by_semantics', '''
>>> import lazyllm
>>> from lazyllm.tools import MarkdownSplitter
>>> splitter = MarkdownSplitter(keep_trace=True, keep_headers=True, keep_lists=True, keep_code_blocks=True, keep_tables=True, keep_images=True, keep_links=True)
>>> md_text = '# Hello, world!\n## Hello, world!\n### Hello, world!\n### Hello, world!'
>>> splits = splitter.split_markdown_by_semantics(md_text)
>>> print(splits)
''')

add_english_doc('rag.transform.code.CodeSplitter', '''
A code splitter that splits code text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    chunk_overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    filetype (Optional[str]): The file type to split. Defaults to None.
    **kwargs: Additional parameters passed to the splitter.
''')

add_chinese_doc('rag.transform.code.CodeSplitter', '''
一个代码拆分器，负责根据文件类型进行路由选择不同的拆分器。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int): 控制并行处理的线程/进程数量。
    filetype (Optional[str]): 要拆分的文件类型。
    **kwargs: 传递给拆分器的额外参数。
''')

add_example('rag.transform.code.CodeSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import CodeSplitter
>>> splitter = CodeSplitter(filetype='python')
''')

add_english_doc('rag.transform.code.CodeSplitter.from_language', '''
Load the language splitter by filetype if not specified in CodeSplitter initialization.

Args:
    filetype (str): The file type to split.
**Returns:**\n
    _LanguageSplitterBase: The language splitter.
''')

add_chinese_doc('rag.transform.code.CodeSplitter.from_language', '''
根据文件类型加载语言拆分器，如果未在CodeSplitter初始化中指定。

Args:
    filetype (str): 要拆分的文件类型。

**Returns:**\n
    _LanguageSplitterBase: 语言拆分器。
''')

add_example('rag.transform.code.CodeSplitter.from_language', '''
>>> import lazyllm
>>> from lazyllm.tools import CodeSplitter
>>> splitter = CodeSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10)
>>> splitter = splitter.from_language('python')
>>> print(splitter)
''')

add_english_doc('rag.transform.code.CodeSplitter.split_text', '''
Split the code text into chunks.

Args:
    text (str): The text to split.
    metadata_size (int): The size of the metadata.
''')

add_chinese_doc('rag.transform.code.CodeSplitter.split_text', '''
拆分代码文本为块。

Args:
    text (str): 要拆分的文本。
    metadata_size (int): 元数据的尺寸。
''')

add_example('rag.transform.code.CodeSplitter.split_text', '''
>>> import lazyllm
>>> from lazyllm.tools import CodeSplitter
>>> splitter = CodeSplitter(filetype='python')
>>> text = 'print("Hello, World!")'
>>> chunks = splitter.split_text(text)
>>> print(chunks)
''')

add_english_doc('rag.transform.code.CodeSplitter.register_splitter', '''
Register a language splitter.

Args:
    filetype (str): The file type to split.
    splitter_class (Type[_LanguageSplitterBase]): The language splitter class.
''')

add_chinese_doc('rag.transform.code.CodeSplitter.register_splitter', '''
注册一个语言拆分器。

Args:
    filetype (str): 要拆分的文件类型。
    splitter_class (Type[_LanguageSplitterBase]): 语言拆分器类。
''')

add_example('rag.transform.code.CodeSplitter.register_splitter', '''
>>> import lazyllm
>>> from lazyllm.tools import CodeSplitter
>>> CodeSplitter.register_splitter('python', PythonSplitter)
''')

add_english_doc('rag.transform.code.CodeSplitter.get_supported_filetypes', '''
Get the supported file types for CodeSplitter.

**Returns:**\n
    List[str]: The supported file types.
''')

add_chinese_doc('rag.transform.code.CodeSplitter.get_supported_filetypes', '''
获取CodeSplitter支持的文件类型。

**Returns:**\n
    List[str]: 支持的文件类型。
''')

add_example('rag.transform.code.CodeSplitter.get_supported_filetypes', '''
>>> import lazyllm
>>> from lazyllm.tools import CodeSplitter
>>> print(CodeSplitter.get_supported_filetypes())
''')

add_english_doc('rag.transform.code.HTMLSplitter', '''
A HTML splitter that splits HTML text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    keep_sections (bool): Whether to keep the sections in the split text. Defaults to False.
    keep_tags (bool): Whether to keep the tags in the split text. Defaults to False.
    **kwargs: Additional parameters passed to the splitter.
''')

add_chinese_doc('rag.transform.code.HTMLSplitter', '''
一个HTML拆分器，负责拆分HTML文本的语义。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int): 控制并行处理的线程/进程数量。
    keep_sections (bool): 是否保留sections在拆分后的文本中。默认为False。
    keep_tags (bool): 是否保留tags在拆分后的文本中。默认为False。
    **kwargs: 传递给拆分器的额外参数。
''')

add_example('rag.transform.code.HTMLSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import HTMLSplitter
>>> splitter = HTMLSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10, keep_sections=True, keep_tags=True)
''')

add_english_doc('rag.transform.code.JSONSplitter', '''
A JSON splitter that splits JSON text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    compact_output (bool): Whether to compact the output. Defaults to True.\
    **kwargs: Additional parameters passed to the splitter.
''')

add_chinese_doc('rag.transform.code.JSONSplitter', '''
一个JSON拆分器，负责拆分JSON文本的语义。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int): 控制并行处理的线程/进程数量。
    compact_output (bool): 是否压缩输出。默认为True。
    **kwargs: 传递给拆分器的额外参数。
''')

add_example('rag.transform.code.JSONSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import JSONSplitter
>>> splitter = JSONSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10, compact_output=True)
>>> print(splitter)
''')

add_english_doc('rag.transform.code.JSONLSplitter', '''
A JSONL splitter that splits JSONL text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    compact_output (bool): Whether to compact the output. Defaults to True.
''')

add_chinese_doc('rag.transform.code.JSONLSplitter', '''
一个JSONL拆分器，负责拆分JSONL文本的语义。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量。
    compact_output (bool): 是否压缩输出。默认为True。
''')

add_example('rag.transform.code.JSONLSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import JSONLSplitter
>>> splitter = JSONLSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10, compact_output=True)
>>> print(splitter)
''')

add_english_doc('rag.transform.code.JSONLSplitter.split_text', '''
Split the JSONL text into chunks.

Args:
    text (str): The text to split.
    metadata_size (int): The size of the metadata.
''')

add_chinese_doc('rag.transform.code.JSONLSplitter.split_text', '''
拆分JSONL文本为块。

Args:
    text (str): 要拆分的文本。
    metadata_size (int): 元数据的尺寸。
''')

add_example('rag.transform.code.JSONLSplitter.split_text', '''
>>> import lazyllm
>>> from lazyllm.tools import JSONLSplitter
>>> splitter = JSONLSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10, compact_output=True)
>>> text = '{"name": "John", "age": 30}\n{"name": "Jane", "age": 25}'
>>> chunks = splitter.split_text(text)
>>> print(chunks)
''')

add_english_doc('rag.transform.code.YAMLSplitter', '''
A YAML splitter that splits YAML text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    compact_output (bool): Whether to compact the output. Defaults to True.
''')

add_chinese_doc('rag.transform.code.YAMLSplitter', '''
一个YAML拆分器，负责拆分YAML文本的语义。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量。
    compact_output (bool): 是否压缩输出。默认为True。
''')

add_example('rag.transform.code.YAMLSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import YAMLSplitter
>>> splitter = YAMLSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10, compact_output=True)
>>> print(splitter)
''')

add_english_doc('rag.transform.code.GeneralCodeSplitter', '''
A general code splitter that splits code text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    filetype (str): The file type to split. Defaults to 'code'.
''')

add_chinese_doc('rag.transform.code.GeneralCodeSplitter', '''
一个通用代码拆分器，负责拆分代码文本的语义。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量。
    filetype (str): 要拆分的文件类型。
''')

add_example('rag.transform.code.GeneralCodeSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import GeneralCodeSplitter
>>> splitter = GeneralCodeSplitter(chunk_size=1024, chunk_overlap=100, num_workers=10, filetype='code')
>>> print(splitter)
''')

add_english_doc('rag.transform.code.XMLSplitter', '''
A XML splitter that splits XML text by semantics.

Args:
    chunk_size (int): The size of the chunk after splitting.
    overlap (int): The length of the overlapping content between two adjacent chunks.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    keep_trace (bool): Whether to keep the trace in the split text. Defaults to False.
    keep_tags (bool): Whether to keep the tags in the split text. Defaults to False.
''')

add_chinese_doc('rag.transform.code.XMLSplitter', '''
一个XML拆分器，负责拆分XML文本的语义。

Args:
    chunk_size (int): 拆分之后的块大小
    overlap (int): 相邻两个块之间重合的内容长度
    num_workers (int):控制并行处理的线程/进程数量。
    keep_trace (bool): 是否保留拆分文本中的trace。
    keep_tags (bool): 是否保留拆分文本中的tags。
''')

add_example('rag.transform.code.XMLSplitter', '''
>>> import lazyllm
>>> from lazyllm.tools import XMLSplitter
>>> splitter = XMLSplitter(chunk_size=1024, overlap=100, num_workers=10, keep_trace=True, keep_tags=True)
>>> print(splitter)
''')

add_english_doc('rag.transform.base.Rule', '''
A rule to apply to the nodes.

Args:
    name (str): The name of the rule.
    match (Callable): The function to match the nodes.
    apply (Callable): The function to apply to the nodes.
    priority (int): The priority of the rule.
    metadata (Dict[str, Any]): The metadata of the rule.
''')

add_chinese_doc('rag.transform.base.Rule', '''
一个规则，用于应用到节点。

Args:
    name (str): 规则的名称。
    match (Callable): 匹配节点的函数。
    apply (Callable): 应用节点的函数。
    priority (int): 规则的优先级。
    metadata (Dict[str, Any]): 规则的元数据。
''')

add_english_doc('rag.transform.base.Rule.build', '''
Build a rule from a pattern string or a predicate function.

Args:
    name (str): The name of the rule.
    rule (Union[str, Callable[[Any], bool]]): The pattern string or the predicate function.
    apply (Callable[[Any, 'Rule'], Any]): The function to apply to the nodes.
''')

add_chinese_doc('rag.transform.base.Rule.build', '''
从模式字符串或谓词函数构建一个规则。

Args:
    name (str): 规则的名称。
    rule (Union[str, Callable[[Any], bool]]): 模式字符串或谓词函数。
    apply (Callable[[Any, 'Rule'], Any]): 应用节点的函数。
''')

add_example('rag.transform.base.Rule.build', '''
>>> import lazyllm
>>> from lazyllm.tools import Rule
>>> rule = Rule.build(name='rule1', rule=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)
>>> print(rule)
''')

add_english_doc('rag.transform.base.RuleSet', '''
A set of rules to apply to the nodes.

Args:
    rules (List[Rule]): The rules to apply to the nodes.
''')

add_chinese_doc('rag.transform.base.RuleSet', '''
一个规则集，用于应用到节点。

Args:
    rules (List[Rule]): 要应用的规则。
''')

add_example('rag.transform.base.RuleSet', '''
>>> import lazyllm
>>> from lazyllm.tools import RuleSet
>>> rules = RuleSet([Rule(name='rule1', rule=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)])
>>> print(rules)
''')

add_english_doc('rag.transform.base.RuleSet.add', '''
Add a rule to the rule set.

Args:
    *rules (Rule): The rules to add to the rule set.
''')

add_chinese_doc('rag.transform.base.RuleSet.add', '''
添加一个规则到规则集。

Args:
    *rules (Rule): 要添加的规则。
''')

add_example('rag.transform.base.RuleSet.add', '''
>>> import lazyllm
>>> from lazyllm.tools import RuleSet
>>> rules = RuleSet([Rule(name='rule1', rule=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)])
>>> rules.add(Rule(name='rule2', rule=lambda n: n.text.endswith('World'), apply=lambda n, r: n))
>>> print(rules)
''')

add_english_doc('rag.transform.base.RuleSet.extend', '''
Extend the rule set with another rule set.

Args:
    rules (RuleSet): The rule set to extend.
''')

add_chinese_doc('rag.transform.base.RuleSet.extend', '''
扩展规则集。

Args:
    rules (RuleSet): 要扩展的规则集。
''')

add_example('rag.transform.base.RuleSet.extend', '''
>>> import lazyllm
>>> from lazyllm.tools import RuleSet
>>> rules = RuleSet([Rule(name='rule1', rule=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)])
>>> rules.extend(RuleSet([Rule(name='rule2', rule=lambda n: n.text.endswith('World'), apply=lambda n, r: n)]))
>>> print(rules)
''')

add_english_doc('rag.transform.base.RuleSet.first', '''
Get the first rule that matches the data.

Args:
    data (Any): The data to match.
''')
add_chinese_doc('rag.transform.base.RuleSet.first', '''
获取第一个匹配数据的规则。
''')

add_example('rag.transform.base.RuleSet.first', '''
>>> import lazyllm
>>> from lazyllm.tools import RuleSet
>>> rules = RuleSet([Rule(name='rule1', rule=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)])
>>> print(rules.first(Node(text='Hello World')))
''')

add_english_doc('rag.transform.base.RuleSet.all', '''
Get all the rules that match the data.

Args:
    data (Any): The data to match.
''')

add_chinese_doc('rag.transform.base.RuleSet.all', '''
获取所有匹配数据的规则。

Args:
    data (Any): 要匹配的数据。
''')

add_example('rag.transform.base.RuleSet.all', '''
>>> import lazyllm
>>> from lazyllm.tools import RuleSet
>>> rules = RuleSet([Rule(name='rule1', rule=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)])
>>> print(rules.all(Node(text='Hello World')))
''')

add_english_doc('rag.transform.base.RuleSet.filter', '''
Filter the rule set with a predicate function.

Args:
    predicate (Callable[[Rule], bool]): The predicate function to filter the rule set.
''')

add_chinese_doc('rag.transform.base.RuleSet.filter', '''
过滤规则集。

Args:
    predicate (Callable[[Rule], bool]): 过滤规则集的谓词函数。
''')

add_english_doc('rag.transform.layout.LayoutNodeParser', '''
A layout node parser that parses layout nodes by semantics.

Args:
    rules (RuleSet): The rules to apply to the nodes.
    group_by (Callable): The function to group the nodes.
    post_process (Callable): The function to post process the nodes.
    sort_by (Callable): The function to sort the nodes.
    return_trace (bool): Whether to return the trace of the nodes.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_chinese_doc('rag.transform.layout.LayoutNodeParser', '''
一个布局节点解析器，负责解析布局节点。

Args:
    rules (RuleSet): 要应用的规则。
    group_by (Callable): 分组节点的函数。
    post_process (Callable): 后处理节点的函数。
    sort_by (Callable): 排序节点的函数。
    return_trace (bool): 是否返回节点的跟踪信息。
    **kwargs: 传递给转换函数的额外参数。
''')

add_example('rag.transform.layout.LayoutNodeParser', '''
>>> import lazyllm
>>> from lazyllm.tools import LayoutNodeParser
>>> parser = LayoutNodeParser(rules=RuleSet([Rule(name='rule1', match=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)]))
>>> parser(nodes)
''')

add_english_doc('rag.transform.treebuilder.TreeBuilderParser', '''
A tree builder parser that builds a tree from the nodes.

Args:
    rules (RuleSet): The rules to apply to the nodes.
    get_level (Callable): The function to get the level of the nodes.
    is_valid_child (Callable): The function to check if a node is a valid child.
    return_trace (bool): Whether to return the trace of the nodes.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_chinese_doc('rag.transform.treebuilder.TreeBuilderParser', '''
一个树构建器，负责构建一个树。

Args:
    rules (RuleSet): 要应用的规则。
    get_level (Callable): 获取节点层级的函数。
    is_valid_child (Callable): 检查节点是否为有效子节点的函数。
    return_trace (bool): 是否返回节点的跟踪信息。
    **kwargs: 传递给转换函数的额外参数。
''')

add_example('rag.transform.treebuilder.TreeBuilderParser', '''
>>> import lazyllm
>>> from lazyllm.tools import TreeBuilderParser
>>> parser = TreeBuilderParser(rules=RuleSet([Rule(name='rule1', match=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)]))
>>> parser(nodes)
''')

add_english_doc('rag.transform.treefixer.TreeFixerParser', '''
A tree fixer parser that fixes a tree from the nodes.

Args:
    rules (RuleSet): The rules to apply to the nodes.
    skip_level_under (int): The level to skip the nodes.
    extra_patterns (List[Tuple[str, str]]): The extra patterns to apply to the nodes.
    return_trace (bool): Whether to return the trace of the nodes.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_chinese_doc('rag.transform.treefixer.TreeFixerParser', '''
一个树修复器，负责修复一个树。

Args:
    rules (RuleSet): 要应用的规则。
    skip_level_under (int): 跳过节点层级的函数。
    extra_patterns (List[Tuple[str, str]]): 额外的模式。
    return_trace (bool): 是否返回节点的跟踪信息。
    **kwargs: 传递给转换函数的额外参数。
''')

add_example('rag.transform.treefixer.TreeFixerParser', '''
>>> import lazyllm
>>> from lazyllm.tools import TreeFixerParser
>>> parser = TreeFixerParser(rules=RuleSet([Rule(name='rule1', match=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)]))
>>> parser(nodes)
''')

add_english_doc('rag.transform.contentfilter.ContentFiltParser', '''
A content filter parser that filters the nodes by the rules.

Args:
    rules (RuleSet): The rules to apply to the nodes.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_chinese_doc('rag.transform.contentfilter.ContentFiltParser', '''
一个内容过滤器解析器，负责过滤节点。

Args:
    rules (RuleSet): 要应用的规则。
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_example('rag.transform.contentfilter.ContentFiltParser', '''
>>> import lazyllm
>>> from lazyllm.tools import ContentFiltParser
>>> parser = ContentFiltParser(rules=RuleSet([Rule(name='rule1', match=lambda n: n.text.startswith('Hello'), apply=lambda n, r: n)]))
>>> parser(nodes)
''')

add_english_doc('rag.transform.groupby.GroupNodeParser', '''
A group node parser that groups the nodes by the rules.

Args:
    max_length (int): The maximum length of the nodes.
    merge_title (bool): Whether to merge the title of the nodes.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
    return_trace (bool): Whether to return the trace of the nodes.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_chinese_doc('rag.transform.groupby.GroupNodeParser', '''
一个组节点解析器，负责组节点。

Args:
    max_length (int): 节点的最大长度。
    merge_title (bool): 是否合并节点的标题。
    num_workers (int): 控制并行处理的线程/进程数量。
    return_trace (bool): 是否返回节点的跟踪信息。
    **kwargs: 传递给转换函数的额外参数。
''')

add_example('rag.transform.groupby.GroupNodeParser', '''
>>> import lazyllm
>>> from lazyllm.tools import GroupNodeParser
>>> parser = GroupNodeParser(max_length=1024, merge_title=True, num_workers=10)
>>> parser(nodes)
''')

add_english_doc('rag.transform.groupby.GroupNodeParser.process', '''
Process nodes with optional match and miss handlers.

Args:
    nodes (List[Any]): The nodes to process.
    on_match (Optional[Callable]): The function to handle matched nodes.
    on_miss (Optional[Callable]): The function to handle missed nodes.
''')

add_chinese_doc('rag.transform.groupby.GroupNodeParser.process', '''
处理节点，可选匹配和缺失处理。

Args:
    nodes (List[Any]): 要处理的节点。
    on_match (Optional[Callable]): 处理匹配节点的函数。
    on_miss (Optional[Callable]): 处理缺失节点的函数。
''')

add_example('rag.transform.groupby.GroupNodeParser.process', '''
>>> import lazyllm
>>> from lazyllm.tools import GroupNodeParser
>>> parser = GroupNodeParser(max_length=1024, merge_title=True, num_workers=10)
>>> nodes = parser.process(nodes, on_match=lambda n, mr, ctx: mr[1], on_miss=lambda n, ctx: n)
''')

add_english_doc('rag.transform.base.NodeTransform', '''
Processes document nodes in batch, supporting both single-threaded and multi-threaded modes.

Args:
    num_workers (int): Controls whether multi-threading is enabled (enabled when >0).
    rules (RuleSet): The rules to apply to the nodes.
    return_trace (bool): Whether to return the trace of the nodes.
    **kwargs: Additional parameters passed to the transformation function.
''')

add_chinese_doc('rag.transform.base.NodeTransform', '''
批量处理文档节点，支持单线程/多线程模式。

Args:
    num_workers (int)：控制是否启用多线程（>0 时启用, 默认为0）。
    rules (RuleSet): 要应用的规则。
    return_trace (bool): 是否返回节点的跟踪信息。
    **kwargs: 传递给转换函数的额外参数。
''')

add_example('rag.transform.base.NodeTransform', '''
>>> import lazyllm
>>> from lazyllm.tools import NodeTransform
>>> node_tran = NodeTransform(num_workers=num_workers)
>>> doc = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
>>> nodes = node_tran.batch_forward(doc, "word_split")
''')

add_english_doc('rag.transform.base.NodeTransform.batch_forward', '''
Process documents in batch with node group transformation.

Args:
    documents (Union[DocNode, List[DocNode]]): Input node(s) to process.
    node_group (str): Target transformation group name.
    **kwargs: Additional transformation parameters.
''')

add_chinese_doc('rag.transform.base.NodeTransform.batch_forward', '''
批量处理文档节点并生成指定组的子节点。

Args:
    documents (Union[DocNode, List[DocNode]]): 待处理的输入节点（单个或列表）。
    node_group (str): 目标转换组名称。
    **kwargs: 额外转换参数。
''')

add_english_doc('rag.transform.base.NodeTransform.forward', '''
[Abstract] Core transformation logic to implement.

Args:
    nodes (Union[List[DocNode], DocNode]): Input document node(s).
    **kwargs: Implementation-specific parameters.
''')

add_chinese_doc('rag.transform.base.NodeTransform.forward', '''
[抽象方法] 需要子类实现的核心转换逻辑。

Args:
    nodes (Union[List[DocNode], DocNode]): 输入文档节点(s)。
    **kwargs: 实现相关的参数。
''')

add_english_doc('rag.transform.base.NodeTransform.with_name', '''
Set transformer name with optional copying.

Args:
    name (Optional[str]): New name for the transformer.
    copy (bool): Whether to return a copy. Default True.
''')

add_chinese_doc('rag.transform.base.NodeTransform.with_name', '''
设置转换器名称）。

Args:
    name (Optional[str]): 转换器的新名称。
    copy (bool): 是否返回副本，默认为True。
''')

add_english_doc('rag.transform.base.NodeTransform.process', '''
Process nodes with optional match and miss handlers.

Args:
    nodes (List[Any]): The nodes to process.
    on_match (Optional[Callable]): The function to handle matched nodes.
    on_miss (Optional[Callable]): The function to handle missed nodes.
''')

add_chinese_doc('rag.transform.base.NodeTransform.process', '''
处理节点，可选匹配和缺失处理。

Args:
    nodes (List[Any]): 要处理的节点。
    on_match (Optional[Callable]): 处理匹配节点的函数。
    on_miss (Optional[Callable]): 处理缺失节点的函数。
''')

add_example('rag.transform.base.NodeTransform.process', '''
>>> import lazyllm
>>> from lazyllm.tools import NodeTransform
>>> node_tran = NodeTransform(num_workers=num_workers)
>>> doc = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
>>> nodes = node_tran.batch_forward(doc, "word_split")
>>> nodes = node_tran.process(nodes, on_match=lambda n, mr, ctx: mr[1], on_miss=lambda n, ctx: n)
''')

add_english_doc('rag.transform.factory.TransformArgs', '''
A document transformation parameter container for centralized management of processing configurations.

Args:
    f (Union[str, Callable]): Transformation function or registered function name.Can be either a callable function or a string identifier for registered functions.
    trans_node (bool): Whether to transform node types.When True, modifies the document node structure during processing.
    num_workers (int):Controls parallel processing threads.Values >0.
    kwargs (Dict):Additional parameters passed to the transformation function.
    pattern (Union[str, Callable[[str], bool]]):File name/content matching pattern.
''')

add_chinese_doc('rag.transform.factory.TransformArgs', '''
文档转换参数容器，用于统一管理文档处理中的各类配置参数。

Args:
    f(Union[str, Callable]):转换函数或注册的函数名。
    trans_node(bool):是否转换节点类型。
    num_workers (int)：控制是否启用多线程（>0 时启用）。
    kwargs(Dict): 传递给转换函数的额外参数。
    pattern(Union[str, Callable[[str], bool]]):文件名/内容匹配模式。
''')

add_example('rag.transform.factory.TransformArgs', '''
>>> from lazyllm.tools import TransformArgs
>>> args = TransformArgs(f=lambda text: text.lower(),num_workers=4,pattern=r'.*\.md$')
>>>config = {'f': 'parse_pdf','kwargs': {'engine': 'pdfminer'},'trans_node': True}
>>>args = TransformArgs.from_dict(config)
print(args['f'])
print(args.get('unknown'))
''')

add_english_doc('rag.transform.factory.LLMParser', '''
A text summarizer and keyword extractor that is responsible for analyzing the text input by the user and providing concise summaries or extracting relevant keywords based on the requested task.

Args:
    llm (TrainableModule): A trainable module.
    language (str): The language type, currently only supports Chinese (zh) and English (en).
    task_type (str): Currently supports two types of tasks: summary and keyword extraction.
    num_workers (int): Controls the number of threads or processes used for parallel processing.
''')

add_chinese_doc('rag.transform.factory.LLMParser', '''
一个文本摘要和关键词提取器，负责分析用户输入的文本，并根据请求任务提供简洁的摘要或提取相关关键词。

Args:
    llm (TrainableModule): 可训练的模块
    language (str): 语言种类，目前只支持中文（zh）和英文（en）
    task_type (str): 目前支持两种任务：摘要（summary）和关键词抽取（keywords）。
    num_workers (int): 控制并行处理的线程/进程数量。
''')

add_example('rag.transform.factory.LLMParser', '''
>>> from lazyllm import TrainableModule
>>> from lazyllm.tools.rag import LLMParser
>>> llm = TrainableModule("internlm2-chat-7b")
>>> summary_parser = LLMParser(llm, language="en", task_type="summary")
''')

add_english_doc('rag.transform.factory.LLMParser.forward', '''
Perform the set task on the specified document.

Args:
    node (DocNode): The document on which the extraction task needs to be performed.
''')

add_chinese_doc('rag.transform.factory.LLMParser.forward', '''
在指定的文档上执行设定的任务。

Args:
    node (DocNode): 需要执行抽取任务的文档。
''')

add_example('rag.transform.factory.LLMParser.forward', '''
>>> import lazyllm
>>> from lazyllm.tools import LLMParser
>>> llm = lazyllm.TrainableModule("internlm2-chat-7b").start()
>>> m = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
>>> summary_parser = LLMParser(llm, language="en", task_type="summary")
>>> keywords_parser = LLMParser(llm, language="en", task_type="keywords")
>>> documents = lazyllm.Document(dataset_path="/path/to/your/data", embed=m, manager=False)
>>> rm = lazyllm.Retriever(documents, group_name='CoarseChunk', similarity='bm25', topk=6)
>>> doc_nodes = rm("test")
>>> summary_result = summary_parser.forward(doc_nodes[0])
>>> keywords_result = keywords_parser.forward(doc_nodes[0])
''')

# FuncNodeTransform
add_english_doc('rag.transform.factory.FuncNodeTransform', '''
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

add_chinese_doc('rag.transform.factory.FuncNodeTransform', '''
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

add_example('rag.transform.factory.FuncNodeTransform', '''
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
add_english_doc('rag.transform.factory.FuncNodeTransform.forward', '''
Transform a document node using the wrapped user-defined function.

This method applies the user-defined function to either the text content of the node (when trans_node=False) or the node itself (when trans_node=True).

Args:
    node (DocNode): The document node to be transformed.
    **kwargs: Additional keyword arguments passed to the transformation function.

**Returns:**\n
- List[Union[str, DocNode]]: The transformed results, which can be either strings or DocNode objects depending on the function implementation.
''')

add_chinese_doc('rag.transform.factory.FuncNodeTransform.forward', '''
使用包装的用户自定义函数转换文档节点。

此方法将用户自定义函数应用于节点的文本内容（当 trans_node=False 时）或节点本身（当 trans_node=True 时）。

Args:
    node (DocNode): 要转换的文档节点。
    **kwargs: 传递给转换函数的额外关键字参数。

**Returns:**\n
- List[Union[str, DocNode]]: 转换结果，根据函数实现可以是字符串或 DocNode 对象。
''')

add_english_doc('rag.transform.factory.AdaptiveTransform', '''\
A flexible document transformation system that applies different transforms based on document patterns.

AdaptiveTransform allows you to define multiple transformation strategies and automatically selects the appropriate one based on the document's file path or custom pattern matching. This is particularly useful when you have different types of documents that require different processing approaches.

Args:
    transforms (Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]]): A list of transform configurations or a single transform configuration. 
    num_workers (int, optional): Number of worker threads for parallel processing. Defaults to 0.
''')

add_chinese_doc('rag.transform.factory.AdaptiveTransform', '''\
一个灵活的文档转换系统，根据文档模式应用不同的转换策略。

AdaptiveTransform允许您定义多种转换策略，并根据文档的文件路径或自定义模式匹配自动选择适当的转换方法。当您有不同类型的文档需要不同处理方法时，这特别有用。

Args:
    transforms (Union[List[Union[TransformArgs, Dict]], Union[TransformArgs, Dict]]): 转换配置列表或单个转换配置。
    num_workers (int, optional): 并行处理的工作线程数。默认为0。
''')

add_example('rag.transform.factory.AdaptiveTransform', '''\
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
>>> results1 = adaptive.forward(doc1)
>>> print(f"文档1转换结果: {len(results1)} 个块")
>>> for i, result in enumerate(results1):
...     print(f"  块 {i+1}: {result.text}")
>>> results2 = adaptive.forward(doc2)
>>> print(f"文档2转换结果: {len(results2)} 个块")
>>> for i, result in enumerate(results2):
...     print(f"  块 {i+1}: {result.text}")      
''')

add_english_doc('rag.transform.factory.AdaptiveTransform.forward', '''\
Transform a document using the appropriate transformation strategy based on pattern matching.

This method evaluates each transform configuration in order and applies the first one that matches the document's path pattern. The matching logic supports both glob patterns and custom callable functions.

Args:
    document (DocNode): The document node to be transformed.
    **kwargs: Additional keyword arguments passed to the transform function.

**Returns:**\n
- List[Union[str, DocNode]]: A list of transformed results (strings or DocNode objects).
''')

add_chinese_doc('rag.transform.factory.AdaptiveTransform.forward', '''\
根据模式匹配使用适当的转换策略转换文档。

此方法按顺序评估每个转换配置，并应用第一个匹配文档路径模式的转换。匹配逻辑支持glob模式和自定义可调用函数。

Args:
    document (DocNode): 要转换的文档节点。
    **kwargs: 传递给转换函数的附加关键字参数。

**Returns:**\n
- List[Union[str, DocNode]]: 转换结果列表（字符串或DocNode对象）。
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

add_chinese_doc('WebModule', '''\
WebModule是LazyLLM为开发者提供的基于Web的交互界面。在初始化并启动一个WebModule之后，开发者可以从页面上看到WebModule背后的模块结构，并将Chatbot组件的输入传输给自己开发的模块进行处理。
模块返回的结果和日志会直接显示在网页的“处理日志”和Chatbot组件上。除此之外，WebModule支持在网页上动态加入Checkbox或Text组件用于向模块发送额外的参数。
WebModule页面还提供“使用上下文”，“流式输出”和“追加输出”的Checkbox，可以用来改变页面和后台模块的交互方式。


Args:
    m (Any): 要包装的模型对象，可以是lazyllm.FlowBase子类或其他可调用对象。
    components (Dict[Any, Any], optional): 额外的UI组件配置，默认为空字典。
    title (str, optional): Web页面标题，默认为'对话演示终端'。
    port (Optional[Union[int, range, tuple, list]], optional): 服务端口号或端口范围，默认为20500-20799。
    history (List[Any], optional): 历史会话模块列表，默认为空列表。
    text_mode (Optional[Mode], optional): 文本输出模式（Dynamic/Refresh/Appendix），默认为Dynamic。
    trace_mode (Optional[Mode], optional): 追踪模式参数(已弃用)。
    audio (bool, optional): 是否启用音频输入功能，默认为False。
    stream (bool, optional): 是否启用流式输出，默认为False。
    files_target (Optional[Union[Any, List[Any]]], optional): 文件处理的目标模块，默认为None。
    static_paths (Optional[Union[str, Path, List[Union[str, Path]]]], optional): 静态资源路径，默认为None。
    encode_files (bool, optional): 是否对文件路径进行编码处理，默认为False。
    share (bool, optional): 是否生成可分享的公共链接，默认为False。
''')

add_english_doc('WebModule', '''\
WebModule is a web-based interactive interface provided by LazyLLM for developers. After initializing and starting
a WebModule, developers can see structure of the module they provides behind the WebModule, and transmit the input
of the Chatbot component to their modules. The results and logs returned by the module will be displayed on the
“Processing Logs” and Chatbot component on the web page. In addition, Checkbox or Text components can be added
programmatically to the web page for additional parameters to the background module. Meanwhile, The WebModule page
provides Checkboxes of “Use Context,” “Stream Output,” and “Append Output,” which can be used to adjust the
interaction between the page and the module behind.

Args:
    m (Any): The model object to wrap, can be a lazyllm.FlowBase subclass or other callable object.
    components (Dict[Any, Any], optional): Additional UI component configurations, defaults to empty dict.
    title (str, optional): Web page title, defaults to 'Dialogue Demo Terminal'.
    port (Optional[Union[int, range, tuple, list]], optional): Service port number or port range, defaults to 20500-20799.
    history (List[Any], optional): List of historical session modules, defaults to empty list.
    text_mode (Optional[Mode], optional): Text output mode (Dynamic/Refresh/Appendix), defaults to Dynamic.
    trace_mode (Optional[Mode], optional): Deprecated trace mode parameter.
    audio (bool, optional): Whether to enable audio input functionality, defaults to False.
    stream (bool, optional): Whether to enable streaming output, defaults to False.
    files_target (Optional[Union[Any, List[Any]]], optional): Target module for file processing, defaults to None.
    static_paths (Optional[Union[str, Path, List[Union[str, Path]]]], optional): Static resource paths, defaults to None.
    encode_files (bool, optional): Whether to encode file paths, defaults to False.
    share (bool, optional): Whether to generate a shareable public link, defaults to False.
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

**Returns:**\n
- gr.Blocks: 构建好的 Gradio 页面对象，可用于 launch 启动 Web 服务。
''')

add_english_doc('WebModule.init_web', '''\
Initialize the Web UI page.
This method uses Gradio to build the interactive chat interface and binds all components to the appropriate logic. It supports session selection, streaming output, context toggling, multimodal input, and control tools. The method returns the constructed Gradio Blocks object.

Args:
    component_descs (List[Tuple]): A list of component descriptors. Each element is a 5-tuple
        (module, group_name, name, component_type, value), e.g. ('MyModule', 'GroupA', 'use_cache', 'Checkbox', True).

**Returns:**\n
- gr.Blocks: The constructed Gradio UI object, which can be launched via `.launch()`.
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

add_chinese_doc('ParameterExtractor.choose_prompt', '''
根据参数描述内容选择合适的提示模板（Prompt）。

此方法会检查传入的参数描述字符串中是否包含中文字符：

- 如果包含中文字符，则返回中文提示模板 `ch_parameter_extractor_prompt`；
- 如果不包含中文字符，则返回英文提示模板 `en_parameter_extractor_prompt`。

Args:
    prompt (str): 参数描述字符串，用于判断使用中文或英文提示模板。

**Returns:**\n
- str: 对应语言的提示模板（Prompt）。
''')

add_english_doc('ParameterExtractor.choose_prompt', '''
Selects the appropriate prompt template based on the content of the parameter descriptions.

This method checks whether the input parameter description string contains any Chinese characters:

- If Chinese characters are present, returns the Chinese prompt template `ch_parameter_extractor_prompt`.
- Otherwise, returns the English prompt template `en_parameter_extractor_prompt`.

Args:
    prompt (str): Parameter description string used to determine whether to use the Chinese or English prompt template.

**Returns:**\n
- str: Prompt template in the corresponding language.
''')
add_chinese_doc('ParameterExtractor.check_int_value', """\
检查并转换整数值。

确保整型参数的值正确转换为int类型。

Args:
    res (dict): 包含参数值的字典
""")

add_english_doc('ParameterExtractor.check_int_value', """\
Check and convert integer values.

Ensure integer parameter values are correctly converted to int type.

Args:
    res (dict): Dictionary containing parameter values
""")
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

**Returns:**\n
- str: The selected prompt template string (either Chinese or English version).
''')

add_chinese_doc('QustionRewrite.choose_prompt', '''
根据输入提示的语言选择合适的提示模板。

此方法分析输入提示字符串并确定使用中文还是英文提示模板。它检查提示字符串中的每个字符，如果任何字符落在中文字符Unicode范围内（\\u4e00-\\u9fff），则返回中文提示模板；否则返回英文提示模板。

Args:
    prompt (str): 要分析语言检测的输入提示字符串。

**Returns:**\n
- str: 选定的提示模板字符串（中文或英文版本）。
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

add_chinese_doc("MongoDBManager", """\
MongoDBManager是与MongoB数据库进行交互的专用工具。它提供了检查连接，获取数据库连接对象，执行查询的方法。

Args:
   user (str): MongoDB用户名
    password (str): MongoDB密码
    host (str): MongoDB服务器地址
    port (int): MongoDB服务器端口
    db_name (str): 数据库名称
    collection_name (str): 集合名称
    **kwargs: 额外配置参数，包括：
        - options_str (str): 连接选项字符串
        - collection_desc_dict (dict): 集合描述字典
""")

add_english_doc("MongoDBManager", """\
MongoDBManager is a specialized tool for interacting with MongoB databases.
It provides methods to check the connection, obtain the database connection object, and execute query.

Args:
   user (str): MongoDB username
    password (str): MongoDB password
    host (str): MongoDB server address
    port (int): MongoDB server port
    db_name (str): Database name
    collection_name (str): Collection name
    **kwargs: Additional configuration parameters including:
        - options_str (str): Connection options string
        - collection_desc_dict (dict): Collection description dictionary
""")

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


add_chinese_doc("MongoDBManager.get_client", """\
这是一个上下文管理器，它创建并返回一个数据库会话连接对象，并在使用完成后自动关闭会话。
使用方式例如：

with mongodb_manager.get_client() as client:
    all_dbs = client.list_database_names()

**Returns:**\n
- pymongo.MongoClient: 连接 MongoDB 数据库的对象
""")

add_english_doc("MongoDBManager.get_client", """\
This is a context manager that creates a database session, yields it for use, and closes the session when done.
Usage example:

with mongodb_manager.get_client() as client:
    all_dbs = client.list_database_names()

**Returns:**\n
- pymongo.MongoClient: MongoDB client used to connect to MongoDB database
""")

add_chinese_doc("MongoDBManager.check_connection", """\
检查当前MongoDBManager的连接状态。

**Returns:**\n
- DBResult: DBResult.status 连接成功(True), 连接失败(False)。DBResult.detail 包含失败信息
""")

add_english_doc("MongoDBManager.check_connection", """\
Check the current connection status of the MongoDBManager.

**Returns:**\n
- DBResult: DBResult.status True if the connection is successful, False if it fails. DBResult.detail contains failure information.
""")

add_chinese_doc("MongoDBManager.set_desc", """\
对于MongoDBManager搭配LLM使用自然语言查询的文档集设置其必须的关键字描述。注意，查询需要用到的关系字都必须提供，因为MonoDB无法像SQL数据库一样获得表结构信息

Args:
    schema_desc_dict (dict): 文档集的关键字描述
""")

add_english_doc("MongoDBManager.set_desc", """\
When using MongoDBManager with LLM to query documents in natural language, set descriptions for the necessary keywords. Note that all relevant keywords needed for queries must be provided because MongoDB cannot obtain like structural information like a SQL database.

Args:
    tables_desc_dict (dict): descriptive comment for documents
""")

add_chinese_doc("SqlCall", """\
SqlCall 是一个扩展自 ModuleBase 的类,提供了使用语言模型(LLM)生成和执行 SQL 查询的接口。
它设计用于与 SQL 数据库交互,从语言模型的响应中提取 SQL 查询,执行这些查询,并返回结果或解释。

Args:
    llm: 用于生成和解释 SQL 查询及解释的大语言模型。
    sql_manager (DBManager): 数据库管理器实例，包含数据库连接和描述信息
    sql_examples (str, optional): SQL示例字符串，用于提示工程。默认为空字符串
    sql_post_func (Callable, optional): 对生成的SQL语句进行后处理的函数。默认为 ``None``
    use_llm_for_sql_result (bool, optional): 是否使用LLM来解释SQL执行结果。默认为 ``True``
    return_trace (bool, optional): 是否返回执行跟踪信息。默认为 ``False``
""")

add_english_doc("SqlCall", """\
SqlCall is a class that extends ModuleBase and provides an interface for generating and executing SQL queries using a language model (LLM).
It is designed to interact with a SQL database, extract SQL queries from LLM responses, execute those queries, and return results or explanations.

Args:
    llm: A language model to be used for generating and interpreting SQL queries and explanations.
    sql_manager (DBManager): Database manager instance containing connection and description information
    sql_examples (str, optional): SQL example strings for prompt engineering. Defaults to empty string
    sql_post_func (Callable, optional): Function for post-processing generated SQL statements. Defaults to ``None``
    use_llm_for_sql_result (bool, optional): Whether to use LLM to explain SQL execution results. Defaults to ``True``
    return_trace (bool, optional): Whether to return execution trace information. Defaults to ``False``
""")

add_example("SqlCall", """\
    >>> # First, run SqlManager example
    >>> import lazyllm
    >>> from lazyllm.tools import SQLManger, SqlCall
    >>> sql_tool = SQLManger("personal.db")
    >>> sql_llm = lazyllm.OnlineChatModule(model="gpt-4o", source="openai", base_url="***")
    >>> sql_call = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=True)
    >>> print(sql_call("去年一整年销售额最多的员工是谁?"))
""")

add_english_doc('SqlCall.sql_query_promt_hook', '''\
Hook to prepare the prompt inputs for generating a database query from user input.

Args:
    input (Union[str, List, Dict[str, str], None]): The user's natural language query.
    history (List[Union[List[str], Dict[str, Any]]]): Conversation history.
    tools (Union[List[Dict[str, Any]], None]): Available tool descriptions.
    label (Union[str, None]): Optional label for the prompt.

**Returns:**\n
- Tuple: A tuple containing the formatted prompt dict (with current_date, db_type, desc, user_query), history, tools, and label.
''')

add_chinese_doc('SqlCall.sql_query_promt_hook', '''\
为从用户输入生成数据库查询准备 prompt 的 hook。

Args:
    input (Union[str, List, Dict[str, str], None]): 用户的自然语言查询。
    history (List[Union[List[str], Dict[str, Any]]]): 会话历史。
    tools (Union[List[Dict[str, Any]], None]): 可用工具描述。
    label (Union[str, None]): 可选标签。

**Returns:**\n
- Tuple: 包含格式化后的 prompt 字典（包括 current_date、db_type、desc、user_query）、history、tools 和 label。
''')

add_english_doc('SqlCall.sql_explain_prompt_hook', '''\
Hook to prepare the prompt for explaining the execution result of a database query.

Args:
    input (Union[str, List, Dict[str, str], None]): A list containing the query and its result.
    history (List[Union[List[str], Dict[str, Any]]]): Conversation history.
    tools (Union[List[Dict[str, Any]], None]): Available tool descriptions.
    label (Union[str, None]): Optional label for the prompt.

**Returns:**\n
- Tuple: A tuple containing the formatted prompt dict (history_info, desc, query, result, explain_query), history, tools, and label.
''')

add_chinese_doc('SqlCall.sql_explain_prompt_hook', '''\
为解释数据库查询执行结果准备 prompt 的 hook。

Args:
    input (Union[str, List, Dict[str, str], None]): 包含查询和结果的列表。
    history (List[Union[List[str], Dict[str, Any]]]): 会话历史。
    tools (Union[List[Dict[str, Any]], None]): 可用工具描述。
    label (Union[str, None]): 可选标签。

**Returns:**\n
- Tuple: 包含格式化后的 prompt 字典（history_info、desc、query、result、explain_query）、history、tools 和 label。
''')

add_english_doc('SqlCall.extract_sql_from_response', '''\
Extract SQL (or MongoDB pipeline) statement from the raw LLM response.

Args:
    str_response (str): Raw text returned by the LLM which may contain code fences.

**Returns:**\n
- tuple[bool, str]: A tuple where the first element indicates whether extraction succeeded, and the second is the cleaned or original content. If sql_post_func is provided, it is applied to the extracted content.
''')

add_chinese_doc('SqlCall.extract_sql_from_response', '''\
从原始 LLM 响应中提取 SQL（或 MongoDB pipeline）语句。

Args:
    str_response (str): LLM 返回的原始文本，可能包含代码块。

**Returns:**\n
- tuple[bool, str]: 第一个元素表示是否成功提取，第二个是清洗后的或原始内容。如果提供了 sql_post_func，则会应用于提取结果。
''')

add_english_doc('SqlCall.create_from_document', '''\
Build a `SqlCall` tool directly from a `Document` that already has a bound `SchemaExtractor`. It reuses the extractor’s NL2SQL `SqlManager` and LLM so you can generate and execute SQL against the document’s registered schemas.

Args:
    document (Document): A Document instance with an attached SchemaExtractor (schema-aware Document).
    llm (optional): Override LLM for SQL generation/answering; defaults to the extractor’s LLM.
    sql_examples (str, optional): Few-shot examples appended to the schema description to guide SQL generation.
    sql_post_func (Callable, optional): Post-processor applied to the extracted SQL/pipeline before execution.
    use_llm_for_sql_result (bool, optional): Whether to ask the LLM to explain query results; default True.
    return_trace (bool, optional): Whether to return pipeline trace; default False.

**Returns:**\n
- SqlCall: Configured SqlCall instance tied to the Document’s schema tables.
''')

add_chinese_doc('SqlCall.create_from_document', '''\
基于已绑定 SchemaExtractor 的 Document 创建 SqlCall，复用其 NL2SQL SqlManager 和 LLM，可直接面向文档注册的 schema 生成/执行 SQL。

Args:
    document (Document): 具备 SchemaExtractor 的文档实例。
    llm (optional): 覆盖用于 SQL 生成/结果说明的 LLM，默认复用文档的 LLM。
    sql_examples (str, optional): 追加在 schema 描述后的 few-shot 示例，指导 SQL 生成。
    sql_post_func (Callable, optional): 对提取的 SQL/管道做后处理的函数。
    use_llm_for_sql_result (bool, optional): 是否用 LLM 解释查询结果，默认 True。
    return_trace (bool, optional): 是否返回流水线 trace，默认 False。

**Returns:**\n
- SqlCall: 绑定到该 Document schema 表的 SqlCall 实例。
''')

# ---------------------------------------------------------------------------- #

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

add_english_doc('rag.doc_node.RichDocNode', '''\
A specialized document node for aggregating multiple paragraph nodes with individual metadata, to keep each document with only one root node.

RichDocNode extends DocNode to wrap multiple child nodes (typically paragraphs) returned by readers. It preserves the full document text content while allowing each child node to maintain its own metadata. When combined with RichTransform, the original DocNode instances (with their metadata) can be recovered.

Args:
    nodes (List[DocNode]): The list of paragraph nodes to aggregate. Each node's text is combined into the content.
    uid (Optional[str]): Unique identifier for the document node. If not provided, a UUID will be automatically generated.
    group (Optional[str]): The group name this node belongs to. Used for organizing and filtering nodes.
    embedding (Optional[Dict[str, List[float]]]): Pre-computed embeddings. Keys are embedding model names, values are embedding vectors.
    parent (Optional[DocNode]): Parent node in the document hierarchy. Used for building document trees.
    metadata (Optional[Dict[str, Any]]): Additional metadata associated with the node.
    global_metadata (Optional[Dict[str, Any]]): Global metadata that applies to all nodes in the document.

Notes:
    - Commonly returned by PDF readers when multiple nodes are produced from a single document, as root node.
    - The original paragraph nodes are stored internally and can be accessed via RichTransform.
    - Preserves the entire document text as a list of paragraph texts in the content field.
''')

add_chinese_doc('rag.doc_node.RichDocNode', '''\
用于聚合多个带有独立元数据的段落节点的专用文档节点，以保持每个文档进有一个root node。

RichDocNode继承自DocNode，用于封装reader返回的多个子节点（通常是段落）。它保留完整的文档文本内容，同时允许每个子节点维护自己的元数据。结合RichTransform使用时，可以恢复出原始的DocNode实例（带有元信息）。

Args:
    nodes (List[DocNode]): 要聚合的段落节点列表。每个节点的文本会被合并到content中。
    uid (Optional[str]): 文档节点的唯一标识符。如果未提供，将自动生成UUID。
    group (Optional[str]): 此节点所属的组名。用于组织和过滤节点。
    embedding (Optional[Dict[str, List[float]]]): 预计算的嵌入。键是嵌入模型名称，值是嵌入向量。
    parent (Optional[DocNode]): 文档层次结构中的父节点。用于构建文档树。
    metadata (Optional[Dict[str, Any]]): 与节点关联的附加元数据。
    global_metadata (Optional[Dict[str, Any]]): 适用于文档中所有节点的全局元数据。

Notes:
    - 通常由PDF reader在单个文档产生多个节点时返回，作为root node。
    - 原始段落节点存储在内部，可通过RichTransform访问恢复。
    - 以段落文本列表的形式在content字段中保留整篇文档文本。
''')

add_english_doc('rag.doc_node.JsonDocNode', '''\
A specialized document node for handling JSON content in RAG systems.

JsonDocNode extends DocNode to provide functionality for storing and processing JSON data (dictionaries or lists). It automatically serializes JSON content to string format and supports custom formatting through a JsonFormatter.

Args:
    uid (Optional[str]): Unique identifier for the document node. If not provided, a UUID will be automatically generated.
    content (Optional[Union[Dict[str, Any], List[Any]]]): The JSON content to store. Can be a dictionary or a list.
    group (Optional[str]): The group name this node belongs to. Used for organizing and filtering nodes.
    embedding (Optional[Dict[str, List[float]]]): Pre-computed embeddings. Keys are embedding model names, values are embedding vectors.
    parent (Optional[DocNode]): Parent node in the document hierarchy. Used for building document trees.
    metadata (Optional[Dict[str, Any]]): Additional metadata associated with the node.
    global_metadata (Optional[Dict[str, Any]]): Global metadata that applies to all nodes in the document.
    formatter (JsonFormatter, optional): A formatter for custom JSON content representation. Used when retrieving content for embedding.

Notes:
    - The text property returns the JSON content serialized as a string.
    - When a formatter is provided, get_content() uses it for embedding-mode output, only vectorize the specified fields, joined by newline.
''')

add_chinese_doc('rag.doc_node.JsonDocNode', '''\
用于处理RAG系统中JSON内容的专用文档节点。

JsonDocNode继承自DocNode，提供存储和处理JSON数据（字典或列表）的功能。它自动将JSON内容序列化为字符串格式，并支持通过JsonFormatter进行自定义格式化。

Args:
    uid (Optional[str]): 文档节点的唯一标识符。如果未提供，将自动生成UUID。
    content (Optional[Union[Dict[str, Any], List[Any]]]): 要存储的JSON内容。可以是字典或列表。
    group (Optional[str]): 此节点所属的组名。用于组织和过滤节点。
    embedding (Optional[Dict[str, List[float]]]): 预计算的嵌入。键是嵌入模型名称，值是嵌入向量。
    parent (Optional[DocNode]): 文档层次结构中的父节点。用于构建文档树。
    metadata (Optional[Dict[str, Any]]): 与节点关联的附加元数据。
    global_metadata (Optional[Dict[str, Any]]): 适用于文档中所有节点的全局元数据。
    formatter (JsonFormatter, optional): 用于自定义JSON内容表示的格式化器。在获取用于嵌入的内容时使用。

Notes:
    - text属性返回序列化为字符串的JSON内容。
    - 当提供formatter时，get_content()在向量化模式下使用它进行输出格式化，仅向量化指定的字段，使用换行符连接。
''')

add_english_doc('rag.doc_node.TreeDocNode', '''\
A document node representing a tree structure with explicit direct children links.

TreeDocNode extends DocNode to support tree-like document hierarchies by maintaining a list of direct children. It is commonly used when the parent-child relationships in a document need to be explicitly tracked and navigated.

Args:
    *args: Positional arguments passed to DocNode.__init__.
    **kwargs: Keyword arguments passed to DocNode.__init__ (e.g., uid, content, group, embedding, parent, metadata, global_metadata, text).

Notes:
    - Direct children are stored in `direct_children_in_tree` and can be manipulated via `add_child`.
    - The `from_doc_node` class method can convert an existing DocNode into a TreeDocNode without losing data.
''')

add_chinese_doc('rag.doc_node.TreeDocNode', '''\
表示带有显式直接子节点链接的树结构的文档节点。

TreeDocNode继承自DocNode，通过维护直接子节点列表来支持树状文档层次结构。通常用于需要显式跟踪和导航文档父子关系的场景。

Args:
    *args: 传递给 DocNode.__init__ 的位置参数。
    **kwargs: 传递给 DocNode.__init__ 的关键字参数（例如 uid、content、group、embedding、parent、metadata、global_metadata、text）。

Notes:
    - 直接子节点存储在 `direct_children_in_tree` 中，可通过 `add_child` 方法进行操作。
    - `from_doc_node` 类方法可以将现有的 DocNode 转换为 TreeDocNode，而不会丢失数据。
''')

add_english_doc('rag.doc_node.TreeDocNode.from_doc_node', '''\
Convert an existing DocNode into a TreeDocNode.

If the input node is already a TreeDocNode, it is returned unchanged. Otherwise, a shallow copy is made, its class is switched to TreeDocNode, and `direct_children_in_tree` is initialized to an empty list.

Args:
    node (DocNode): The document node to convert.

Returns:
    TreeDocNode: The resulting tree document node.
''')

add_chinese_doc('rag.doc_node.TreeDocNode.from_doc_node', '''\
将现有的 DocNode 转换为 TreeDocNode。

如果输入节点已经是 TreeDocNode，则原样返回。否则，将创建一个浅拷贝，切换其类为 TreeDocNode，并将 `direct_children_in_tree` 初始化为空列表。

Args:
    node (DocNode): 要转换的文档节点。

Returns:
    TreeDocNode: 转换后的树文档节点。
''')

add_english_doc('rag.doc_node.TreeDocNode.add_child', '''\
Add a direct child node to this tree node.

Args:
    child (TreeDocNode): The child node to append.

Returns:
    None
''')

add_chinese_doc('rag.doc_node.TreeDocNode.add_child', '''\
向当前树节点添加一个直接子节点。

Args:
    child (TreeDocNode): 要添加的子节点。

Returns:
    None
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

# agent/functionCall.py
add_chinese_doc('rag.web.DocWebModule', """\
文档Web界面模块，继承自ModuleBase，提供基于Web的文档管理交互界面。

Args:
    doc_server (ServerModule): 文档服务模块实例，提供后端API支持
    title (str, optional): 界面标题，默认为"文档管理演示终端"
    port (optional): 服务端口号或端口范围。默认为 ``None``（使用20800-20999范围）
    history (optional): 初始聊天历史记录，默认为 ``None``
    text_mode (optional): 文本处理模式，默认为``None``(动态模式)
    trace_mode (optional): 追踪模式，默认为``None``(刷新模式)

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
    title (str, optional): Interface title. Defaults to 'Document Management Demo Terminal'
    port (optional): Service port number or port range. Defaults to ``None`` (uses range 20800-20999)
    history (optional): History record list. Defaults to ``None``
    text_mode (optional): Text display mode. Defaults to ``None`` (uses dynamic mode)
    trace_mode (optional): Trace mode. Defaults to ``None`` (uses refresh mode)


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
Block current thread waiting for web service to run.

This method blocks the calling thread until the web service is explicitly stopped.

''')

add_chinese_doc('rag.web.DocWebModule.wait', '''\
阻塞当前线程等待Web服务运行。

该方法会阻塞调用线程，直到Web服务被显式停止。

''')

add_english_doc('rag.web.DocWebModule.stop', '''\
Stops the web interface service and releases related resources.

''')

add_chinese_doc('rag.web.DocWebModule.stop', '''\
停止Web界面服务并释放相关资源。

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

**Returns:**\n
- dict: HTTP 请求头字典。
''')

add_english_doc("rag.web.WebUi.basic_headers", '''
Generate standard HTTP headers.

Args:
    content_type (bool): Whether to include Content-Type in the headers (default: True).

**Returns:**\n
- dict: Dictionary of HTTP headers.
''')

add_chinese_doc("rag.web.WebUi.muti_headers", '''
生成多部分表单的HTTP请求头。
用于文件上传等需要multipart/form-data格式的请求。

**Returns:**\n
- Dict: 返回包含accept头部的HTTP请求头字典。
''')

add_english_doc("rag.web.WebUi.muti_headers", '''
Generates multipart form HTTP request headers.
Used for requests requiring multipart/form-data format such as file uploads.

**Returns:**\n
- Dict: Returns HTTP request header dictionary containing accept header.
''')

add_chinese_doc("rag.web.WebUi.post_request", '''
发送 POST 请求。

Args:
    url (str): 请求地址。
    data (dict): 请求数据，将被转为 JSON。

**Returns:**\n
- dict: 响应结果的 JSON。
''')

add_english_doc("rag.web.WebUi.post_request", '''
Send a POST request.

Args:
    url (str): Target request URL.
    data (dict): Request data (will be serialized as JSON).

**Returns:**\n
- dict: JSON response from the server.
''')

add_chinese_doc("rag.web.WebUi.get_request", '''
发送 GET 请求。

Args:
    url (str): 请求地址。

**Returns:**\n
- dict: 响应结果的 JSON。
''')

add_english_doc("rag.web.WebUi.get_request", '''
Send a GET request.

Args:
    url (str): Target request URL.

**Returns:**\n
- dict: JSON response from the server.
''')

add_chinese_doc("rag.web.WebUi.new_group", '''
创建新的文件分组。

Args:
    group_name (str): 分组名称。

**Returns:**\n
- str: 创建结果的提示信息。
''')

add_english_doc("rag.web.WebUi.new_group", '''
Create a new file group.

Args:
    group_name (str): Name of the new group.

**Returns:**\n
- str: Server message about the creation result.
''')

add_chinese_doc("rag.web.WebUi.delete_group", '''
删除指定的文件分组。

Args:
    group_name (str): 分组名称。

**Returns:**\n
- str: 删除结果信息。
''')

add_english_doc("rag.web.WebUi.delete_group", '''
Delete a specific file group.

Args:
    group_name (str): Name of the group to delete.

**Returns:**\n
- str: Server message about the deletion.
''')

add_chinese_doc("rag.web.WebUi.list_groups", '''
获取所有知识库分组列表。
向后台API发送请求，获取当前所有的知识库分组信息。

**Returns:**\n
- List: 返回分组名称列表。
''')

add_english_doc("rag.web.WebUi.list_groups", '''
Gets all knowledge base group list.
Sends request to backend API to get all current knowledge base group information.

**Returns:**\n
- List: Returns group name list.
''')

add_chinese_doc("rag.web.WebUi.upload_files", '''
向指定分组上传文件。

Args:
    group_name (str): 分组名称。
    override (bool): 是否覆盖已存在的文件（默认 True）。

**Returns:**\n
- Any: 后端返回的上传结果数据。
''')

add_english_doc("rag.web.WebUi.upload_files", '''
Upload files to a specified group.

Args:
    group_name (str): Name of the group.
    override (bool): Whether to override existing files (default: True).

**Returns:**\n
- Any: Data returned by the backend.
''')

add_chinese_doc("rag.web.WebUi.list_files_in_group", '''
列出指定分组下的所有文件。

Args:
    group_name (str): 分组名称。

**Returns:**\n
- List: 文件信息列表。
''')

add_english_doc("rag.web.WebUi.list_files_in_group", '''
List all files within a specific group.

Args:
    group_name (str): Name of the group.

**Returns:**\n
- List: List of file information.
''')

add_chinese_doc("rag.web.WebUi.delete_file", '''
从指定分组中删除文件。

Args:
    group_name (str): 分组名称。
    file_ids (List[str]): 要删除的文件 ID 列表。

**Returns:**\n
- str: 删除结果提示。
''')

add_english_doc("rag.web.WebUi.delete_file", '''
Delete specific files from a group.

Args:
    group_name (str): Name of the group.
    file_ids (List[str]): IDs of files to delete.

**Returns:**\n
- str: Deletion result message.
''')

add_chinese_doc("rag.web.WebUi.gr_show_list", '''
以 Gradio 表格的形式展示字符串列表。

Args:
    str_list (List): 字符串或子项列表。
    list_name (Union[str, List]): 表头名称或列名列表。

**Returns:**\n
- gr.DataFrame: Gradio 表格组件。
''')

add_english_doc("rag.web.WebUi.gr_show_list", '''
Display a list of strings as a Gradio DataFrame.

Args:
    str_list (List): List of strings or rows.
    list_name (Union[str, List]): Column name(s) for the table.

**Returns:**\n
- gr.DataFrame: Gradio DataFrame component.
''')

add_chinese_doc("rag.web.WebUi.create_ui", '''
构建包含多个标签页的Gradio界面，提供以下功能：
    - 分组列表：查看所有分组信息
    - 上传文件：选择分组并上传文件
    - 分组文件列表：查看指定分组中的文件
    - 删除文件：从分组中删除指定文件

**Returns:**\n
- gr.Blocks: 完整的 Gradio UI 应用实例。
''')

add_english_doc("rag.web.WebUi.create_ui", '''
Builds Gradio interface with multiple tabs, providing the following functionalities:
    - Group List: View all group information
    - Upload Files: Select group and upload files
    - Group File List: View files in specified group
    - Delete Files: Delete specified files from group

**Returns:**\n
- gr.Blocks: A complete Gradio application instance.
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

根据传入的参数执行查询操作，返回匹配的文档节点列表。
**注意:** 当前方法为接口占位，子类需要实现具体逻辑。

Args:
    *args: 任意位置参数，用于查询条件。
    **kwargs: 任意关键字参数，用于查询条件。
''')

add_english_doc('rag.index_base.IndexBase.query', '''\
Execute a query over the index.

Performs a query based on the given arguments and returns matching document nodes.
**Note:** This method is a placeholder and should be implemented by subclasses.

Args:
    *args: Positional arguments for the query.
    **kwargs: Keyword arguments for the query.
''')

add_chinese_doc('StreamCallHelper', '''\
流式调用辅助类，用于将阻塞调用包装为生成器形式，逐步返回执行结果。

Args:
    impl (Callable): 需要流式执行的函数或可调用对象。
    interval (float): 轮询队列的时间间隔，单位为秒，默认为0.1。
''')

add_english_doc('StreamCallHelper', '''\
Helper class for streaming function calls, wrapping a blocking callable into a generator that yields results incrementally.

Args:
    impl (Callable): The function or callable to execute in streaming mode.
    interval (float): Time interval (in seconds) to poll the internal queue. Defaults to 0.1.
''')

add_chinese_doc('rag.LazyLLMStoreBase', '''\
向量存储基类，定义了存储层的通用接口规范，所有具体的存储实现（如 Chroma、Milvus 等）需继承并实现该类。
''')

add_english_doc('rag.LazyLLMStoreBase', '''\
Base class for vector storage, defining the common interface specification.
All concrete storage implementations (e.g., Chroma, Milvus) must inherit and implement this class.
''')

add_chinese_doc('rag.LazyLLMStoreBase.connect', '''\
建立与存储后端的连接。

Args:
    *args: 可变位置参数。
    **kwargs: 可变关键字参数。
''')

add_english_doc('rag.LazyLLMStoreBase.connect', '''\
Establish connection to the storage backend.

Args:
    *args: Variable positional arguments.
    **kwargs: Variable keyword arguments.
''')

add_chinese_doc('rag.LazyLLMStoreBase.upsert', '''\
插入或更新集合中的数据。

Args:
    collection_name (str): 集合名称。
    data (List[dict]): 数据列表，每条为一个记录。
''')

add_english_doc('rag.LazyLLMStoreBase.upsert', '''\
Insert or update data in a collection.

Args:
    collection_name (str): The collection name.
    data (List[dict]): List of records to upsert.
''')

add_chinese_doc('rag.LazyLLMStoreBase.delete', '''\
删除集合中的数据。

Args:
    collection_name (str): 集合名称。
    criteria (dict): 删除条件。
    **kwargs: 额外参数。
''')

add_english_doc('rag.LazyLLMStoreBase.delete', '''\
Delete data from a collection.

Args:
    collection_name (str): The collection name.
    criteria (dict): Conditions for deletion.
    **kwargs: Additional parameters.
''')

add_chinese_doc('rag.LazyLLMStoreBase.get', '''\
根据条件获取集合中的数据。

Args:
    collection_name (str): 集合名称。
    criteria (dict): 过滤条件。
    **kwargs: 额外参数。
''')

add_english_doc('rag.LazyLLMStoreBase.get', '''\
Retrieve data from a collection by criteria.

Args:
    collection_name (str): The collection name.
    criteria (dict): Filter conditions.
    **kwargs: Additional parameters.
''')

add_chinese_doc('rag.LazyLLMStoreBase.search', '''\
执行检索操作，可以基于文本或向量。

Args:
    collection_name (str): 集合名称。
    query (Optional[str]): 文本查询字符串。
    query_embedding (Optional[Union[dict, List[float]]]): 查询向量。
    topk (int): 返回的结果数量，默认为 10。
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): 元数据过滤条件。
    embed_key (Optional[str]): 向量键。
    **kwargs: 额外参数。
''')

add_english_doc('rag.LazyLLMStoreBase.search', '''\
Perform a search operation, supporting both text and vector queries.

Args:
    collection_name (str): The collection name.
    query (Optional[str]): Text query string.
    query_embedding (Optional[Union[dict, List[float]]]): Query vector.
    topk (int): Number of results to return. Defaults to 10.
    filters (Optional[Dict[str, Union[str, int, List, Set]]]): Metadata filter conditions.
    embed_key (Optional[str]): Embedding key.
    **kwargs: Additional parameters.
''')

add_chinese_doc('rag.doc_impl.DocImpl', '''\
文档实现类，用于管理文档处理、存储和检索的核心功能。

Args:
    embed (Dict[str, Callable]): 嵌入函数字典。
    dataset_path (Optional[str]): 数据集目录路径，默认为None。
    doc_files (Optional[str]): 文档文件路径，默认为None。
    kb_group_name (Optional[str]): 知识库组名称，默认为默认组名。
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): 全局元数据描述。
    store (Optional[Union[Dict, LazyLLMStoreBase]]): 存储实例或配置。
    processor (Optional[DocumentProcessor]): 文档处理服务。
    algo_name (Optional[str]): 算法名称。
    display_name (Optional[str]): 显示名称。
    description (Optional[str]): 描述信息。
''')

add_english_doc('rag.doc_impl.DocImpl', '''\
Document implementation class for managing core document processing, storage, and retrieval functionalities.

Args:
    embed (Dict[str, Callable]): Dictionary of embedding functions.
    dataset_path (Optional[str]): Dataset directory path, defaults to None.
    doc_files (Optional[str]): Document files path, defaults to None.
    kb_group_name (Optional[str]): Knowledge base group name, defaults to default group name.
    global_metadata_desc (Dict[str, GlobalMetadataDesc]): Global metadata description.
    store (Optional[Union[Dict, LazyLLMStoreBase]]): Storage instance or configuration.
    processor (Optional[DocumentProcessor]): Document processing service.
    algo_name (Optional[str]): Algorithm name.
    display_name (Optional[str]): Display name.
    description (Optional[str]): Description information.
''')

add_chinese_doc('rag.doc_impl.DocImpl.create_global_node_group', '''\
创建全局节点组。

Args:
    name (str): 节点组名称。
    transform (Union[str, Callable]): 转换函数或名称。
    parent (str): 父节点组名称，默认为LAZY_ROOT_NAME。
    trans_node (Optional[bool]): 是否转换节点，默认为None。
    num_workers (int): 工作线程数，默认为0。
    display_name (Optional[str]): 显示名称。
    group_type (NodeGroupType): 节点组类型。
    **kwargs: 其他参数。
''')

add_english_doc('rag.doc_impl.DocImpl.create_global_node_group', '''\
Create a global node group.

Args:
    name (str): Node group name.
    transform (Union[str, Callable]): Transform function or name.
    parent (str): Parent node group name, defaults to LAZY_ROOT_NAME.
    trans_node (Optional[bool]): Whether to transform node, defaults to None.
    num_workers (int): Number of worker threads, defaults to 0.
    display_name (Optional[str]): Display name.
    group_type (NodeGroupType): Node group type.
    **kwargs: Additional arguments.
''')

add_chinese_doc('rag.doc_impl.DocImpl.create_node_group', '''\
创建局部节点组。

Args:
    name (str): 节点组名称。
    transform (Union[str, Callable]): 转换函数或名称。
    parent (str): 父节点组名称，默认为LAZY_ROOT_NAME。
    trans_node (Optional[bool]): 是否转换节点，默认为None。
    num_workers (int): 工作线程数，默认为0。
    display_name (Optional[str]): 显示名称。
    group_type (NodeGroupType): 节点组类型。
    **kwargs: 其他参数。
''')

add_english_doc('rag.doc_impl.DocImpl.create_node_group', '''\
Create a local node group.

Args:
    name (str): Node group name.
    transform (Union[str, Callable]): Transform function or name.
    parent (str): Parent node group name, defaults to LAZY_ROOT_NAME.
    trans_node (Optional[bool]): Whether to transform node, defaults to None.
    num_workers (int): Number of worker threads, defaults to 0.
    display_name (Optional[str]): Display name.
    group_type (NodeGroupType): Node group type.
    **kwargs: Additional arguments.
''')

add_chinese_doc('rag.doc_impl.DocImpl.register_global_reader', '''\
注册全局文件读取器。

Args:
    pattern (str): 文件模式。
    func (Optional[Callable]): 读取函数，默认为None。

**Returns:**\n
- Optional[Callable]: 装饰器函数或None。
''')

add_english_doc('rag.doc_impl.DocImpl.register_global_reader', '''\
Register a global file reader.

Args:
    pattern (str): File pattern.
    func (Optional[Callable]): Reader function, defaults to None.

**Returns:**\n
- Optional[Callable]: Decorator function or None.
''')

add_chinese_doc('rag.doc_impl.DocImpl.register_index', '''\
注册索引。

Args:
    index_type (str): 索引类型。
    index_cls (IndexBase): 索引类。
    *args: 位置参数。
    **kwargs: 关键字参数。
''')

add_english_doc('rag.doc_impl.DocImpl.register_index', '''\
Register an index.

Args:
    index_type (str): Index type.
    index_cls (IndexBase): Index class.
    *args: Positional arguments.
    **kwargs: Keyword arguments.
''')

add_chinese_doc('rag.doc_impl.DocImpl.add_reader', '''\
添加局部文件读取器。

Args:
    pattern (str): 文件模式。
    func (Optional[Callable]): 读取函数。
''')

add_english_doc('rag.doc_impl.DocImpl.add_reader', '''\
Add a local file reader.

Args:
    pattern (str): File pattern.
    func (Optional[Callable]): Reader function.
''')

add_chinese_doc('rag.doc_impl.DocImpl.activate_group', '''\
激活节点组。

Args:
    group_name (str): 节点组名称。
    embed_keys (List[str]): 嵌入键列表。
''')

add_english_doc('rag.doc_impl.DocImpl.activate_group', '''\
Activate a node group.

Args:
    group_name (str): Node group name.
    embed_keys (List[str]): List of embedding keys.
''')

add_chinese_doc('rag.doc_impl.DocImpl.active_node_groups', '''\
获取当前激活的节点组。

**Returns:**\n
- Dict: 激活的节点组及其嵌入键。
''')

add_english_doc('rag.doc_impl.DocImpl.active_node_groups', '''\
Get currently active node groups.

**Returns:**\n
- Dict: Active node groups and their embedding keys.
''')

add_chinese_doc('rag.doc_impl.DocImpl.retrieve', '''\
检索文档节点。

Args:
    query (str): 查询字符串。
    group_name (str): 节点组名称。
    similarity (str): 相似度计算方法。
    similarity_cut_off (Union[float, Dict[str, float]]): 相似度阈值。
    index (str): 索引类型。
    topk (int): 返回结果数量。
    similarity_kws (dict): 相似度计算参数。
    embed_keys (Optional[List[str]]): 嵌入键列表。
    filters (Optional[Dict]): 过滤条件。
    **kwargs: 其他参数。

**Returns:**\n
- List[DocNode]: 检索到的文档节点列表。
''')

add_english_doc('rag.doc_impl.DocImpl.retrieve', '''\
Retrieve document nodes.

Args:
    query (str): Query string.
    group_name (str): Node group name.
    similarity (str): Similarity calculation method.
    similarity_cut_off (Union[float, Dict[str, float]]): Similarity threshold.
    index (str): Index type.
    topk (int): Number of results to return.
    similarity_kws (dict): Similarity calculation parameters.
    embed_keys (Optional[List[str]]): List of embedding keys.
    filters (Optional[Dict]): Filter conditions.
    **kwargs: Additional arguments.

**Returns:**\n
- List[DocNode]: List of retrieved document nodes.
''')

add_chinese_doc('rag.doc_impl.DocImpl.find', '''\
在指定组中查找节点。

Args:
    nodes (List[DocNode]): 节点列表。
    group (str): 目标组名称。

**Returns:**\n
- List[DocNode]: 找到的节点列表。
''')

add_english_doc('rag.doc_impl.DocImpl.find', '''\
Find nodes in specified group.

Args:
    nodes (List[DocNode]): List of nodes.
    group (str): Target group name.

**Returns:**\n
- List[DocNode]: List of found nodes.
''')

add_chinese_doc('rag.doc_impl.DocImpl.find_parent', '''\
查找父节点。

Args:
    nodes (List[DocNode]): 节点列表。
    group (str): 目标组名称。

**Returns:**\n
- List[DocNode]: 找到的父节点列表。
''')

add_english_doc('rag.doc_impl.DocImpl.find_parent', '''\
Find parent nodes.

Args:
    nodes (List[DocNode]): List of nodes.
    group (str): Target group name.

**Returns:**\n
- List[DocNode]: List of found parent nodes.
''')

add_chinese_doc('rag.doc_impl.DocImpl.find_children', '''\
查找子节点。

Args:
    nodes (List[DocNode]): 节点列表。
    group (str): 目标组名称。

**Returns:**\n
- List[DocNode]: 找到的子节点列表。
''')

add_english_doc('rag.doc_impl.DocImpl.find_children', '''\
Find child nodes.

Args:
    nodes (List[DocNode]): List of nodes.
    group (str): Target group name.

**Returns:**\n
- List[DocNode]: List of found child nodes.
''')

add_chinese_doc('rag.doc_impl.DocImpl.clear_cache', '''\
清除缓存。

Args:
    group_names (Optional[List[str]]): 要清除缓存的组名列表，默认为None表示清除所有缓存。
''')

add_english_doc('rag.doc_impl.DocImpl.clear_cache', '''\
Clear cache.

Args:
    group_names (Optional[List[str]]): List of group names to clear cache for, defaults to None for clearing all cache.
''')

add_chinese_doc('rag.doc_impl.DocImpl.drop_algorithm', '''\
删除当前文档集合的在文档解析服务中注册的算法信息。
''')

add_english_doc('rag.doc_impl.DocImpl.drop_algorithm', '''\
Delete the algorithm information registered in the document parsing service for the current document collection.
''')

add_chinese_doc('rag.doc_node.DocNode.get_children_str', '''\
获取子节点的字符串表示。

**Returns:**\n
- str: 返回一个字符串，表示子节点的字典格式，其中键为组名，值为该组中所有子节点的UID列表。
''')

add_english_doc('rag.doc_node.DocNode.get_children_str', '''\
Get string representation of child nodes.

**Returns:**\n
- str: Returns a string representing a dictionary where keys are group names and values are lists of child node UIDs in that group.
''')

add_chinese_doc('rag.doc_node.DocNode.get_parent_id', '''\
获取父节点的唯一标识符。

**Returns:**\n
- str: 返回父节点的UID，如果没有父节点则返回空字符串。
''')

add_english_doc('rag.doc_node.DocNode.get_parent_id', '''\
Get the unique identifier of the parent node.

**Returns:**\n
- str: Returns the parent node's UID, or an empty string if there is no parent node.
''')

add_chinese_doc('rag.doc_node.DocNode.get_content', '''\
获取节点的内容文本，包含LLM模式的元数据。

**Returns:**\n
- str: 返回节点的文本内容，包含根据LLM模式格式化的元数据信息。
''')

add_english_doc('rag.doc_node.DocNode.get_content', '''\
Get the node's content text with metadata in LLM mode.

**Returns:**\n
- str: Returns the node's text content with formatted metadata information according to LLM mode.
''')

add_chinese_doc('rag.store.hybrid.MapStore', """\
基于SQLite的Map存储类，继承自LazyLLMStoreBase。

提供基于SQLite的数据持久化与BM25全文检索，支持多集合管理和简单查询。

Args:
    uri (Optional[str]): SQLite数据库文件路径，默认为None（内存模式）
    **kwargs: 其他关键字参数

Attributes:
    capability: 存储能力标志，支持所有操作
    need_embedding: 是否需要嵌入向量
    supports_index_registration: 是否支持索引注册

""")

add_english_doc('rag.store.hybrid.MapStore', """\
SQLite-based Map storage class, inherits from LazyLLMStoreBase.

Provides data persistence and BM25 full-text search via SQLite for lightweight use cases.

Args:
    uri (Optional[str]): SQLite database file path, defaults to None (in-memory mode)
    **kwargs: Other keyword arguments

Attributes:
    capability: Storage capability flag, supports all operations
    need_embedding: Whether embedding is required
    supports_index_registration: Whether index registration is supported
""")

add_chinese_doc('rag.store.hybrid.MapStore.connect', """\
连接SQLite数据库并加载数据。

初始化存储连接，创建必要的数据库表和索引，加载现有数据到内存。

Args:
    collections (Optional[List[str]]): 要连接的集合名称列表
    **kwargs: 其他连接参数

Returns:
    None
""")

add_english_doc('rag.store.hybrid.MapStore.connect', """\
Connect to SQLite database and load data.

Initialize storage connection, create necessary database tables and indexes, load existing data into memory.

Args:
    collections (Optional[List[str]]): List of collection names to connect
    **kwargs: Other connection parameters

Returns:
    None
""")

add_chinese_doc('rag.store.hybrid.MapStore.upsert', """\
插入或更新数据。

将数据插入到指定集合中，如果已存在则更新，支持批量操作。

Args:
    collection_name (str): 集合名称
    data (List[dict]): 要插入的数据列表

Returns:
    bool: 操作是否成功
""")

add_english_doc('rag.store.hybrid.MapStore.upsert', """\
Insert or update data.

Insert data into specified collection, update if exists, supports batch operations.

Args:
    collection_name (str): Collection name
    data (List[dict]): Data list to insert

Returns:
    bool: Whether operation succeeded
""")

add_chinese_doc('rag.store.hybrid.MapStore.delete', """\
删除数据。

根据条件删除指定集合中的数据，支持批量删除。

Args:
    collection_name (str): 集合名称
    criteria (Optional[dict]): 删除条件
    **kwargs: 其他删除参数

Returns:
    bool: 操作是否成功
""")

add_english_doc('rag.store.hybrid.MapStore.delete', """\
Delete data.

Delete data from specified collection based on criteria, supports batch deletion.

Args:
    collection_name (str): Collection name
    criteria (Optional[dict]): Delete criteria
    **kwargs: Other delete parameters

Returns:
    bool: Whether operation succeeded
""")

add_chinese_doc('rag.store.hybrid.MapStore.get', """\
查询数据。

根据条件查询指定集合中的数据，支持多种查询条件。

Args:
    collection_name (str): 集合名称
    criteria (Optional[dict]): 查询条件
    **kwargs: 其他查询参数

Returns:
    List[dict]: 查询结果数据列表
""")

add_english_doc('rag.store.hybrid.MapStore.get', """\
Query data.

Query data from specified collection based on criteria, supports multiple query conditions.

Args:
    collection_name (str): Collection name
    criteria (Optional[dict]): Query criteria
    **kwargs: Other query parameters

Returns:
    List[dict]: Query result data list
""")

add_chinese_doc('rag.store.segment.OpenSearchStore', """\
OpenSearch存储类，继承自LazyLLMStoreBase。

提供基于OpenSearch的文档存储和检索功能，支持大规模文档管理和高效查询。

Args:
    uris (List[str]): OpenSearch服务URI列表
    client_kwargs (Optional[Dict]): OpenSearch客户端配置参数
    index_kwargs (Optional[Union[Dict, List]]): 索引配置参数
    **kwargs: 其他关键字参数

Attributes:
    capability: 存储能力标志，支持分段操作
    need_embedding: 是否需要嵌入向量
    supports_index_registration: 是否支持索引注册

""")

add_english_doc('rag.store.OpenSearchStore', """\
OpenSearch storage class, inherits from LazyLLMStoreBase.

Provides document storage and retrieval functionality based on OpenSearch, supports large-scale document management and efficient query.

Args:
    uris (List[str]): OpenSearch service URI list
    client_kwargs (Optional[Dict]): OpenSearch client configuration parameters
    index_kwargs (Optional[Union[Dict, List]]): Index configuration parameters
    **kwargs: Other keyword arguments

Attributes:
    capability: Storage capability flag, supports segment operations
    need_embedding: Whether embedding is needed
    supports_index_registration: Whether index registration is supported


""")

add_chinese_doc('rag.store.segment.OpenSearchStore.connect', """\
连接OpenSearch服务。

初始化OpenSearch客户端连接，配置认证信息。

Args:
    *args: 位置参数
    **kwargs: 关键字参数

Returns:
    None
""")

add_english_doc('rag.store.segment.OpenSearchStore.connect', """\
Connect to OpenSearch service.

Initialize OpenSearch client connection, configure authentication information.

Args:
    *args: Positional arguments
    **kwargs: Keyword arguments

Returns:
    None
""")

add_chinese_doc('rag.store.segment.OpenSearchStore.upsert', """\
插入或更新数据到OpenSearch。

批量插入或更新文档数据到指定集合（索引），支持自动创建索引。

Args:
    collection_name (str): 集合名称（索引名）
    data (List[dict]): 要插入的数据列表

Returns:
    bool: 操作是否成功
""")

add_english_doc('rag.store.segment.OpenSearchStore.upsert', """\
Insert or update data to OpenSearch.

Batch insert or update document data to specified collection (index), supports automatic index creation.

Args:
    collection_name (str): Collection name (index name)
    data (List[dict]): Data list to insert

Returns:
    bool: Whether operation succeeded
""")

add_chinese_doc('rag.store.segment.OpenSearchStore.delete', """\
从OpenSearch删除数据。

根据条件删除指定集合中的数据，支持批量删除和索引删除。

Args:
    collection_name (str): 集合名称（索引名）
    criteria (Optional[dict]): 删除条件
    **kwargs: 其他删除参数

Returns:
    bool: 操作是否成功
""")

add_english_doc('rag.store.segment.OpenSearchStore.delete', """\
Delete data from OpenSearch.

Delete data from specified collection based on criteria, supports batch deletion and index deletion.

Args:
    collection_name (str): Collection name (index name)
    criteria (Optional[dict]): Delete criteria
    **kwargs: Other delete parameters

Returns:
    bool: Whether operation succeeded
""")

add_chinese_doc('rag.store.segment.OpenSearchStore.get', """\
从OpenSearch查询数据。

根据条件查询指定集合中的数据，支持主键查询和复杂条件查询。

Args:
    collection_name (str): 集合名称（索引名）
    criteria (Optional[dict]): 查询条件
    **kwargs: 其他查询参数

Returns:
    List[dict]: 查询结果数据列表
""")

add_english_doc('rag.store.segment.OpenSearchStore.get', """\
Query data from OpenSearch.

Query data from specified collection based on criteria, supports primary key query and complex condition query.

Args:
    collection_name (str): Collection name (index name)
    criteria (Optional[dict]): Query criteria
    **kwargs: Other query parameters

Returns:
    List[dict]: Query result data list
""")

add_english_doc('rag.store.segment.OpenSearchStore.search', """\
Perform vector similarity search with optional metadata filtering.
Args:
    collection_name (str): Collection to search.
    query (Optional[str]): Query string.
    topk (Optional[int]): Number of nearest neighbors.
    filters (Optional[dict]): Metadata filter map.
    kwargs: Other search parameters

**Returns:**\n
- List[dict]: Return matching results list and similarity 'score'.
""")

add_chinese_doc('rag.store.segment.OpenSearchStore.search', """\
执行向量相似度检索，并可按元数据过滤。
Args:
    collection_name (str): 待搜索集合。
    query (Optional[str]): 查询字符串。
    topk (Optional[int]): 返回邻近数量。
    filters (Optional[dict]): 元数据过滤映射。
    kwargs: 其他搜索参数

**Returns:**\n
- List[dict]: 返回匹配结果列表及相似度 'score'。
""")

add_chinese_doc('services.ServerBase', """\
服务器基类，提供任务管理和状态监控的基础功能。

实现多用户任务信息存储、状态轮询检查和线程安全的字典操作。


""")

add_english_doc('services.ServerBase', """\
Server base class, provides basic functionality for task management and status monitoring.

Implements multi-user task information storage, status polling check and thread-safe dictionary operations.

""")


add_chinese_doc('services.ServerBase.authorize_current_user', """\
用户认证授权。

验证用户令牌的有效性，确保只有授权用户可以访问相关资源。

Args:
    Bearer: Bearer令牌字符串

Returns:
    str: 验证通过的令牌

Raises:
    HTTPException: 令牌无效时抛出401异常
""")

add_english_doc('services.ServerBase.authorize_current_user', """\
User authentication and authorization.

Verify the validity of user token, ensure only authorized users can access related resources.

Args:
    Bearer: Bearer token string

Returns:
    str: Verified token

Raises:
    HTTPException: 401 exception when token is invalid
""")

# review/tools/chinese_corrector.py

add_chinese_doc('review.tools.chinese_corrector.get_errors', '''\
比较修正文本和原始文本，找出其中的错误位置和内容。

使用序列匹配算法比较两个文本的差异，返回错误列表，每个错误包含原始字符、修正字符和位置信息。

Args:
    corrected_text (str): 修正后的文本。
    origin_text (str): 原始文本。

Returns:
    list: 错误列表，每个元素为 (orig_char, corr_char, pos) 的元组，其中：
        - orig_char (str): 原始字符，如果是插入错误则为空字符串。
        - corr_char (str): 修正字符，如果是删除错误则为空字符串。
        - pos (int): 错误在原始文本中的位置。
''')

add_english_doc('review.tools.chinese_corrector.get_errors', '''\
Compare corrected text with original text to find error locations and contents.

Uses sequence matching algorithm to compare differences between two texts, returns a list of errors,
each containing original character, corrected character, and position information.

Args:
    corrected_text (str): The corrected text.
    origin_text (str): The original text.

Returns:
    list: List of errors, each element is a tuple (orig_char, corr_char, pos) where:
        - orig_char (str): Original character, empty string if insertion error.
        - corr_char (str): Corrected character, empty string if deletion error.
        - pos (int): Position of error in original text.
''')

add_example(
    'review.tools.chinese_corrector.get_errors',
    """\
    >>> from lazyllm.tools.review.tools.chinese_corrector import get_errors
    >>> errors = get_errors("我喜欢编程", "我喜欢编程成")
    >>> print(errors)
    [('', '成', 6)]
""")

add_chinese_doc('review.tools.chinese_corrector.ChineseCorrector', '''\
中文文本纠错器，使用大语言模型对中文句子进行语法和拼写纠错。

通过配置不同的语言模型，可以对单个句子或批量句子进行纠错，并返回纠错结果和错误详情。

Args:
    llm: 可选，大语言模型实例。如果为None，则使用默认模型。
    base_url (str): 可选，模型服务的基础URL。
    model (str): 可选，使用的模型名称。
    api_key (str): 可选，API密钥，默认为'null'。
    source (str): 模型来源，默认为'openai'。
''')

add_english_doc('review.tools.chinese_corrector.ChineseCorrector', '''\
Chinese text corrector that uses large language models to correct grammar and spelling errors in Chinese sentences.

Can correct single sentences or batches of sentences by configuring different language models,
and returns correction results with error details.

Args:
    llm: Optional, large language model instance. Uses default model if None.
    base_url (str): Optional, base URL for model service.
    model (str): Optional, model name to use.
    api_key (str): Optional, API key, defaults to 'null'.
    source (str): Model source, defaults to 'openai'.
''')

add_chinese_doc('review.tools.chinese_corrector.ChineseCorrector.correct', '''\
对单个中文句子进行语法和拼写纠错。

使用配置的语言模型对输入句子进行纠错，并返回包含原始文本、纠错文本和错误详情的结果字典。

Args:
    sentence (str): 需要纠错的中文句子。
    **kwargs: 其他传递给语言模型的参数，如max_tokens、temperature等。

Returns:
    dict: 包含以下键的字典：
        - source (str): 原始输入句子。
        - target (str): 纠错后的句子。
        - errors (list): 错误列表，每个元素为 (orig_char, corr_char, pos) 的元组。
''')

add_english_doc('review.tools.chinese_corrector.ChineseCorrector.correct', '''\
Correct grammar and spelling errors in a single Chinese sentence.

Uses the configured language model to correct the input sentence and returns a dictionary
containing the original text, corrected text, and error details.

Args:
    sentence (str): The Chinese sentence to correct.
    **kwargs: Additional parameters passed to the language model, such as max_tokens, temperature, etc.

Returns:
    dict: Dictionary containing the following keys:
        - source (str): The original input sentence.
        - target (str): The corrected sentence.
        - errors (list): List of errors, each element is a tuple (orig_char, corr_char, pos).
''')

add_chinese_doc('review.tools.chinese_corrector.ChineseCorrector.correct_batch', '''\
批量对中文句子进行语法和拼写纠错。

使用并行处理对多个句子进行纠错，提高处理效率。返回包含每个句子纠错结果的列表。

Args:
    sentences (list): 需要纠错的中文句子列表。
    batch_size (int): 可选，批处理大小，默认为4。
    concurrency (int): 可选，并发数，默认为2。
    **kwargs: 其他传递给语言模型的参数，如max_tokens、temperature等。

Returns:
    list: 每个元素为包含纠错结果的字典列表，每个字典包含：
        - source (str): 原始输入句子。
        - target (str): 纠错后的句子。
        - errors (list): 错误列表，每个元素为 (orig_char, corr_char, pos) 的元组。
''')

add_english_doc('review.tools.chinese_corrector.ChineseCorrector.correct_batch', '''\
Batch correct grammar and spelling errors in multiple Chinese sentences.

Uses parallel processing to correct multiple sentences efficiently. Returns a list of dictionaries
containing correction results for each sentence.

Args:
    sentences (list): List of Chinese sentences to correct.
    batch_size (int): Optional, batch size, defaults to 4.
    concurrency (int): Optional, concurrency level, defaults to 2.
    **kwargs: Additional parameters passed to the language model, such as max_tokens, temperature, etc.

Returns:
    list: List of dictionaries, each containing correction results with keys:
        - source (str): The original input sentence.
        - target (str): The corrected sentence.
        - errors (list): List of errors, each element is a tuple (orig_char, corr_char, pos).
''')

add_example(
    "review.tools.chinese_corrector.ChineseCorrector",
    """\
    >>> import lazyllm
    >>> from lazyllm.tools.review.tools.chinese_corrector import ChineseCorrector
    >>> corrector = ChineseCorrector()
    >>> result = corrector.correct("我喜欢编程成")
    >>> print(result)
    {'source': '我喜欢编程成', 'target': '我喜欢编程', 'errors': [('成', '', 6)]}
    >>>
    >>> results = corrector.correct_batch(["句子1", "句子2"])
    >>> print(results)
    [{'source': '句子1', 'target': '修正后句子1', 'errors': [...]}, ...]
""")

# RAG-QueryEnhACProcessor
add_chinese_doc('rag.QueryEnhACProcessor', '''\
基于 Aho–Corasick（AC）自动机的查询同义词扩展处理器。

在查询串上匹配词表中的词，将保留的命中词替换为「原词 + 同簇其它词形」的展示形式（首遇簇内词时展开，同句后续同簇命中仅保留原词）。匹配先做**最长无重叠**预选，再由判别器判断是否为独立词边界。若 ``discriminator`` 为 ``None`` 且 AC 仍有命中，会记录警告并不改写查询；判别器推理失败时过滤器返回空命中，同样保持原句。

Args:
    data_source: 词表数据源。可为无参可调用（每次调用返回 ``list[dict]``），或直接为字典列表。每条记录需包含 ``cluster_key``、``word_key`` 所指字段；缺字段项会被跳过。
    discriminator: 词边界判别器。支持 ``OnlineChatModule``、用于 LLM 的 ``TrainableModule``（经 ``prompt``/``formatter`` 链），或已 ``deploy_method(lazyllm.deploy.BertDeploy)`` 且 ``start()`` 的 ``TrainableModule``（序列分类，与 ac-jieba 语义对齐）。``None`` 表示不增强。
    cluster_key (str): 同义词簇 ID 字段名，默认 ``"cluster_id"``。
    word_key (str): 词文字段名，默认 ``"word"``。
    max_retries (int): LLM 整批调用或 BERT 逐条调用失败时的最大重试次数，默认 ``3``。
    prompt_lang (Literal["zh", "en"]): 内置 LLM 提示语言；``discriminator`` 为 ``None`` 或 BERT 部署的 ``TrainableModule`` 时无效。

Methods:
    - ``__call__(queries)``: 对单条字符串或字符串列表做增强。
    - ``get_matches(query)``: 返回经 AC + 边界过滤后的匹配列表。
    - ``update_data_source`` / ``update_discriminator``: 热更新词表或判别器。
''')

add_english_doc('rag.QueryEnhACProcessor', '''\
Query synonym expansion using an Aho–Corasick (AC) automaton.

Matches vocabulary words in the query string, then replaces kept hits with the original word plus other surface forms in the same cluster (first hit per cluster is expanded; later hits in the same cluster keep only the matched word). Candidates are pre-filtered to **longest non-overlapping** matches, then a discriminator decides true word boundaries. If ``discriminator`` is ``None`` but the automaton still matches, a warning is logged and the query is unchanged; if the discriminator fails after retries, enhancement is skipped and the original query is kept.

Args:
    data_source: Vocabulary source: a no-arg callable returning ``list[dict]``, or a list of dicts. Each record must contain the fields named by ``cluster_key`` and ``word_key``; incomplete rows are skipped.
    discriminator: Boundary discriminator: ``OnlineChatModule``, ``TrainableModule`` for LLM (with prompt/formatter chain), or a started ``TrainableModule`` with ``deploy_method(lazyllm.deploy.BertDeploy)`` (sequence classification, same semantics as ac-jieba). ``None`` disables enhancement.
    cluster_key (str): Field name for synonym cluster id, default ``"cluster_id"``.
    word_key (str): Field name for surface word text, default ``"word"``.
    max_retries (int): Max retries for batched LLM calls or per-match BERT calls, default ``3``.
    prompt_lang (Literal["zh", "en"]): Language for built-in LLM prompts; ignored when ``discriminator`` is ``None`` or a BERT-deployed ``TrainableModule``.

Methods:
    - ``__call__(queries)``: Enhance a single string or a list of strings.
    - ``get_matches(query)``: Return matches after AC + boundary filtering.
    - ``update_data_source`` / ``update_discriminator``: Hot-swap vocabulary or discriminator.
''')

add_chinese_doc('rag.QueryEnhACProcessor.update_data_source', '''\
热更新词表数据源并重建 AC 自动机。

Args:
    data_source: 新的数据源，规则与构造参数 ``data_source`` 相同。
''')

add_english_doc('rag.QueryEnhACProcessor.update_data_source', '''\
Replace the vocabulary source and rebuild the AC automaton.

Args:
    data_source: New source, same rules as the constructor ``data_source`` argument.
''')

add_chinese_doc('rag.QueryEnhACProcessor.update_discriminator', '''\
热更新词边界判别器（按新实例类型重新构建内部 LLM / BERT 过滤器）。

Args:
    discriminator: 新的判别器；类型须与构造参数一致，可为 ``None``。
''')

add_english_doc('rag.QueryEnhACProcessor.update_discriminator', '''\
Hot-swap the boundary discriminator (rebuilds the internal LLM / BERT filter from the new instance).

Args:
    discriminator: New discriminator; same supported types as the constructor, or ``None``.
''')

add_chinese_doc('rag.QueryEnhACProcessor.get_matches', '''\
返回经 AC 匹配与边界过滤后的命中列表（与 ac-jieba ``return_matches_only`` 形态对齐）。

Args:
    query (str): 待匹配的查询字符串。

**Returns:**
- ``List[dict]``: 每项含当前 ``word_key``、``cluster_key`` 字段，以及 ``cluster_words``（该簇全部词形列表）。
''')

add_english_doc('rag.QueryEnhACProcessor.get_matches', '''\
Return matches after AC matching and boundary filtering (shape aligned with ac-jieba ``return_matches_only``).

Args:
    query (str): The query string to match against the AC automaton.

**Returns:**
- ``List[dict]``: Each item contains the configured ``word_key`` and ``cluster_key`` fields, plus ``cluster_words`` (all words in that cluster).
''')

add_chinese_doc('rag.QueryEnhACProcessor.__call__', '''\
对查询执行同义词扩展增强。

Args:
    queries (Union[str, List[str]]): 单条查询字符串，或查询字符串列表。

**Returns:**
- ``Union[str, List[str]]``: 与输入同结构——单条入参返回字符串，列表入参返回增强后的字符串列表。
''')

add_english_doc('rag.QueryEnhACProcessor.__call__', '''\
Run synonym expansion on one or more queries.

Args:
    queries (Union[str, List[str]]): A single query string or a list of query strings.

**Returns:**
- ``Union[str, List[str]]``: Same structure as input—a string for a single query, or a list of enhanced strings.
''')

add_example('rag.QueryEnhACProcessor', '''\
>>> import lazyllm
>>> from lazyllm.tools.rag import QueryEnhACProcessor
>>> def vocab():
...     return [
...         {"cluster_id": "C1", "word": "民法"},
...         {"cluster_id": "C1", "word": "civil law"},
...     ]
>>> # LLM discriminator
>>> model = lazyllm.OnlineChatModule()
>>> proc = QueryEnhACProcessor(data_source=vocab, discriminator=model)
>>> out = proc("什么是民法？")
>>> print(out)
>>> # BERT discriminator
>>> bert_m = lazyllm.TrainableModule('your-seq-cls-model', use_model_map=False).deploy_method(
...     lazyllm.deploy.BertDeploy, max_length=128,
... ).start()
>>> proc2 = QueryEnhACProcessor(data_source=vocab, discriminator=bert_m)
>>> out2 = proc2("什么是民法？")
>>> print(out2)
>>> proc.update_data_source(vocab)
>>> proc.update_discriminator(model)
''')

add_chinese_doc('rag.data_loaders.DirectoryReader.signature', '''\
计算当前读取器配置的指纹哈希，用于检测 reader 注册表变更。

将本地和全局 reader 映射序列化为 JSON 后取 SHA-256 前 16 位十六进制字符串。
当任意 reader 被替换或新增时，返回值会发生变化，可用于判断是否需要重新解析文档。

**Returns:**\n
- str: 16 位十六进制指纹字符串。
''')

add_english_doc('rag.data_loaders.DirectoryReader.signature', '''\
Compute a fingerprint hash of the current reader configuration to detect registry changes.

Serialises the local and global reader mappings to JSON and returns the first 16 hex characters
of the SHA-256 digest. The value changes whenever any reader is replaced or added, making it
suitable for deciding whether documents need to be re-parsed.

**Returns:**\n
- str: 16-character hexadecimal fingerprint string.
''')



