# Copyright (c) 2026 LazyAGI. All rights reserved.
# flake8: noqa E501
"""FS module docs: LazyLLMFSBase, CloudFSBufferedFile, CloudFS, CloudFsWatchdog, FeishuFS, ConfluenceFS, NotionFS, GoogleDriveFS, OneDriveFS, YuqueFS, OnesFS, S3FS, ObsidianFS."""
import importlib
import functools

from .. import utils

_add_fs_chinese = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.fs'))
_add_fs_english = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.fs'))
_add_fs_example = functools.partial(
    utils.add_example, module=importlib.import_module('lazyllm.tools.fs'))
_feishu_module = importlib.import_module('lazyllm.tools.fs.supplier.feishu')
_add_feishu_chinese = functools.partial(utils.add_chinese_doc, module=_feishu_module)
_add_feishu_english = functools.partial(utils.add_english_doc, module=_feishu_module)
_notion_module = importlib.import_module('lazyllm.tools.fs.supplier.notion')
_add_notion_chinese = functools.partial(utils.add_chinese_doc, module=_notion_module)
_add_notion_english = functools.partial(utils.add_english_doc, module=_notion_module)

_DOCUMENT_LINK_WORKFLOW_ZH = '''\
文档链接使用规则:
    - 当用户提供当前文档 FS 可解析的浏览器链接、分享链接、provider URI 或平台路径时，优先用当前 FS 的链接解析与读取能力，而不是把它当作普通网页 URL 处理。
    - 对私有或需要鉴权的文档，先解析链接并读取正文，再进行总结、分析、回答或引用展开。
    - 如果当前 FS 无法解析、未授权或没有访问权限，应明确说明不可用或无权限，不要假装已经读取。'''
_DOCUMENT_LINK_WORKFLOW_EN = '''\
Document link usage:
    - When users provide a browser link, shared link, provider URI, or provider path supported by this document FS, prefer this FS's link resolution and reading APIs instead of treating it as a generic web URL.
    - For private or authenticated documents, resolve the link and read the body before summarizing, analyzing, answering, or expanding references.
    - If this FS cannot resolve the link, is unauthorized, or lacks access, say that clearly instead of pretending the document was read.'''
_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH = '''\
飞书/Lark 文档链接规则:
    - 用户提供飞书或 Lark 文档、知识库、wiki 链接时，使用当前飞书/Lark 文档文件系统的链接解析和读取能力，先读取正文再回答。
    - 对私有飞书/Lark 页面，不要先用通用 URL 抓取；若未授权或无访问权限，应明确说明。'''
_FEISHU_DOCUMENT_LINK_WORKFLOW_EN = '''\
Feishu/Lark document link usage:
    - When users provide Feishu or Lark document, wiki, or knowledge-base links, use this Feishu/Lark document filesystem's link resolution and reading APIs to read the body before answering.
    - Do not use generic URL fetching first for private Feishu/Lark pages; clearly report missing authorization or access.'''
_NOTION_DOCUMENT_LINK_WORKFLOW_ZH = '''\
Notion 文档链接规则:
    - 用户提供 Notion 页面 URL、notion:/ URI、页面/数据库/block ID 或路径时，先解析对象，再读取页面正文后回答。
    - 对私有 Notion 页面，不要先用通用 URL 抓取；若 integration 未连接页面、未授权或无访问权限，应明确说明。'''
_NOTION_DOCUMENT_LINK_WORKFLOW_EN = '''\
Notion document link usage:
    - When users provide a Notion page URL, notion:/ URI, page/database/block id, or path, resolve the object first, then read the page body before answering.
    - Do not use generic URL fetching first for private Notion pages; clearly report when the integration is not connected, unauthorized, or lacks access.'''

# LazyLLMFSBase
_add_fs_chinese('LazyLLMFSBase', '''\
云文件系统统一基类，继承 fsspec.AbstractFileSystem，借助 registry 注册各平台实现。混入 CredentialMixin 提供统一的 token 生命周期管理。
子类需实现：_setup_auth、ls、info、_open、_download_range、_upload_data 等；可选实现 rm_file、mkdir。
目录监听逻辑由 CloudFsWatchdog 提供，FS 子类仅需提供必要的 webhook 能力（若有）。

Args:
    token (Any): 认证信息载体，通常为字符串 token，部分子类也可能封装为 dict/tuple 等结构。
    base_url (str, optional): API 或服务根地址。
    asynchronous (bool): 是否启用异步模式，对应 fsspec.AbstractFileSystem.asynchronous，默认 False。
    use_listings_cache (bool): 是否缓存目录列表，对应 fsspec.AbstractFileSystem.use_listings_cache，默认 False。
    skip_instance_cache (bool): 是否跳过实例缓存，对应 fsspec.AbstractFileSystem.skip_instance_cache，默认 False。
    loop (Any, optional): 异步事件循环对象，一般仅在异步环境下需要显式传入。
    auth_strategy (AuthStrategy, optional): 认证策略，决定 token 如何注入请求头或查询参数；未传时默认使用 BearerTokenStrategy。
''')
_add_fs_english('LazyLLMFSBase', '''\
Unified cloud filesystem base; extends fsspec.AbstractFileSystem; implementations registered via registry. Mixes in CredentialMixin for unified token lifecycle management.
Subclasses implement _setup_auth, ls, info, _open, _download_range, _upload_data; optionally rm_file, mkdir.
Directory watching is handled by CloudFsWatchdog; FS subclasses only expose webhook capabilities when supported.

Args:
    token (Any): Auth payload; usually a string token, but some subclasses may encode auth bundles (e.g. dict/tuple) here.
    base_url (str, optional): API or service base URL.
    asynchronous (bool): Whether to enable async mode; forwarded to fsspec.AbstractFileSystem.asynchronous; default False.
    use_listings_cache (bool): Whether to cache directory listings; forwarded to fsspec.AbstractFileSystem.use_listings_cache; default False.
    skip_instance_cache (bool): Whether to skip the filesystem instance cache; forwarded to fsspec.AbstractFileSystem.skip_instance_cache; default False.
    loop (Any, optional): Event loop object for async environments.
    auth_strategy (AuthStrategy, optional): Authentication strategy controlling how the token is injected into request headers or query parameters; defaults to BearerTokenStrategy when omitted.
''')
_add_fs_example('LazyLLMFSBase', '''\
>>> from lazyllm.tools.fs import CloudFS
>>> fs = CloudFS(platform='s3', access_key='xxx', secret_key='yyy')
>>> fs.ls('/')
''')

# LazyLLMFSBase abstract and public methods
_add_fs_chinese('LazyLLMFSBase.ls', '''\
列出路径下的目录项。子类必须实现。

Args:
    path (str): 目录路径。
    detail (bool): 为 True 时返回每项详情（name, size, type, mtime 等）；为 False 时仅返回名称列表。
    **kwargs: 子类扩展参数。

Returns:
    List: detail=True 时为 dict 列表；detail=False 时为 name 列表。
''')
_add_fs_english('LazyLLMFSBase.ls', '''\
List directory entries at path. Must be implemented by subclass.

Args:
    path (str): Directory path.
    detail (bool): If True return list of dicts (name, size, type, mtime etc.); if False return list of names.
    **kwargs: Subclass-specific options.

Returns:
    List: List of dicts if detail=True else list of names.
''')

_add_fs_chinese('LazyLLMFSBase.info', '''\
获取路径对应条目元信息。子类必须实现。

Args:
    path (str): 文件或目录路径。
    **kwargs: 子类扩展参数。

Returns:
    Dict[str, Any]: 至少包含 name, size, type（file/directory），可选 mtime 等。
''')
_add_fs_english('LazyLLMFSBase.info', '''\
Get metadata for the path. Must be implemented by subclass.

Args:
    path (str): File or directory path.
    **kwargs: Subclass-specific options.

Returns:
    Dict[str, Any]: At least name, size, type (file/directory); optional mtime etc.
''')

_add_fs_chinese('LazyLLMFSBase.read', '''\
读取路径对应文件的完整内容并以 UTF-8 字符串返回。支持飞书 URL、~node/~link 等特殊路径。

Args:
    path (str): 文件路径或飞书 URL。

Returns:
    str: 文件文本内容（UTF-8 解码）。
''')
_add_fs_english('LazyLLMFSBase.read', '''\
Read the full content of the file at path and return as a UTF-8 string. Supports Feishu URLs and ~node/~link paths.

Args:
    path (str): File path or Feishu URL.

Returns:
    str: File text content (UTF-8 decoded).
''')

_add_fs_chinese('LazyLLMFSBase.mkdir', '''\
创建目录。基类默认空实现；子类可覆盖。

Args:
    path (str): 目录路径。
    create_parents (bool): 是否递归创建父目录。
    **kwargs: 子类扩展参数。
''')
_add_fs_english('LazyLLMFSBase.mkdir', '''\
Create directory. Base default is no-op; subclass may override.

Args:
    path (str): Directory path.
    create_parents (bool): Whether to create parent directories recursively.
    **kwargs: Subclass-specific options.
''')

_add_fs_chinese('LazyLLMFSBase.makedirs', '''\
递归创建目录，等价于 mkdir(path, create_parents=True)。

Args:
    path (str): 目录路径。
    exist_ok (bool): 已存在是否忽略（当前未使用）。
''')
_add_fs_english('LazyLLMFSBase.makedirs', '''\
Create directory recursively; same as mkdir(path, create_parents=True).

Args:
    path (str): Directory path.
    exist_ok (bool): Ignore if exists (currently unused).
''')

_add_fs_chinese('LazyLLMFSBase.rmdir', '''\
删除空目录。基类默认空实现；子类可覆盖。

Args:
    path (str): 目录路径。
''')
_add_fs_english('LazyLLMFSBase.rmdir', '''\
Remove empty directory. Base default is no-op; subclass may override.

Args:
    path (str): Directory path.
''')

_add_fs_chinese('LazyLLMFSBase.rm_file', '''\
删除单个文件。基类默认抛出 NotImplementedError；子类应实现。

Args:
    path (str): 文件路径。
''')
_add_fs_english('LazyLLMFSBase.rm_file', '''\
Remove a single file. Base raises NotImplementedError; subclass must implement.

Args:
    path (str): File path.
''')

_add_fs_chinese('LazyLLMFSBase.rm', '''\
删除路径；若为目录且 recursive=True 则递归删除后删目录，否则调用 rm_file。

Args:
    path (str): 文件或目录路径。
    recursive (bool): 目录是否递归删除。
''')
_add_fs_english('LazyLLMFSBase.rm', '''\
Remove path; if directory and recursive=True, recursively delete then rmdir; else rm_file.

Args:
    path (str): File or directory path.
    recursive (bool): Whether to recursively delete directory.
''')

_add_fs_chinese('LazyLLMFSBase.exists', '''\
判断路径是否存在。依赖子类实现的 info 或 ls；若底层无此能力则可能由基类/ fsspec 提供。

Args:
    path (str): 文件或目录路径。

Returns:
    bool: 存在为 True，否则 False。
''')
_add_fs_english('LazyLLMFSBase.exists', '''\
Return whether the path exists. Depends on subclass info/ls or fsspec base behavior.

Args:
    path (str): File or directory path.

Returns:
    bool: True if exists, else False.
''')

_add_fs_chinese('LazyLLMFSBase.read_bytes', '''\
将路径对应文件整体读入为字节。路径不存在时抛出 FileNotFoundError；内部使用 open(path, "rb") 读取。

Args:
    path (str): 文件路径。

Returns:
    bytes: 文件完整内容。
''')
_add_fs_english('LazyLLMFSBase.read_bytes', '''\
Read the entire file at path as bytes. Raises FileNotFoundError if path does not exist; uses open(path, "rb") internally.

Args:
    path (str): File path.

Returns:
    bytes: Full file content.
''')

_add_fs_chinese('LazyLLMFSBase.read_file', '''\
将路径对应文件按 UTF-8 解码为字符串。等价于 read_bytes(path).decode("utf-8")。

Args:
    path (str): 文件路径。

Returns:
    str: 文件文本内容。
''')
_add_fs_english('LazyLLMFSBase.read_file', '''\
Read the file at path as UTF-8 string. Same as read_bytes(path).decode("utf-8").

Args:
    path (str): File path.

Returns:
    str: File text content.
''')

_add_fs_chinese('LazyLLMFSBase.write_file', '''\
将字节数据完整写入指定路径。覆盖已存在文件；内部调用 _upload_data。

Args:
    path (str): 远程文件路径。
    data (bytes): 要写入的完整内容。
''')
_add_fs_english('LazyLLMFSBase.write_file', '''\
Write bytes to the given path, overwriting if exists. Uses _upload_data internally.

Args:
    path (str): Remote file path.
    data (bytes): Full content to write.
''')

_add_fs_chinese('LazyLLMFSBase.write', '''\
将文本内容以 UTF-8 编码写入指定路径。覆盖已存在文件；内部调用 _upload_data。

Args:
    path (str): 远程文件路径。
    content (str): 要写入的文本内容。
''')
_add_fs_english('LazyLLMFSBase.write', '''\
Write text content encoded as UTF-8 to the given path, overwriting if exists. Uses _upload_data internally.

Args:
    path (str): Remote file path.
    content (str): Text content to write.
''')

_add_fs_chinese('LazyLLMFSBase.copy', '''\
复制文件或目录到目标路径。各子类调用对应平台的官方接口实现，基类默认抛出 NotImplementedError。

各平台支持情况：

- ObsidianFS：使用 shutil.copy2 / copytree，目录复制需 recursive=True。
- S3FS：使用服务端 copy_object，目录需 recursive=True。
- GoogleDriveFS：使用 files.copy API（仅支持文件，文件夹抛出 NotImplementedError）。
- OneDriveFS：使用 Graph API /copy 接口（异步执行）。
- FeishuFS：使用 Drive v1 /copy 接口（仅支持文件，文件夹抛出 NotImplementedError）。
- FeishuWikiFS：使用 Wiki v2 /copy 接口，支持节点（含子节点）的复制。
- ConfluenceFS：使用 REST API /copy 接口。
- OnesFS / YuqueFS / NotionFS：官方接口不支持，抛出 NotImplementedError。

Args:
    path1 (str): 源路径（文件或目录）。
    path2 (str): 目标路径。
    recursive (bool): 源为目录时是否递归复制子项，默认 False。
    **kwargs: 透传（当前未使用）。
''')
_add_fs_english('LazyLLMFSBase.copy', '''\
Copy file or directory to the target path. Each subclass calls the provider's official API; the base class raises NotImplementedError.

Platform support:

- ObsidianFS: uses shutil.copy2 / copytree; directory copy requires recursive=True.
- S3FS: uses server-side copy_object; directory copy requires recursive=True.
- GoogleDriveFS: uses files.copy API (files only; raises NotImplementedError for folders).
- OneDriveFS: uses Graph API /copy endpoint (async).
- FeishuFS: uses Drive v1 /copy API (files only; raises NotImplementedError for folders).
- FeishuWikiFS: uses Wiki v2 /copy API (supports nodes including children).
- ConfluenceFS: uses REST API /copy endpoint.
- OnesFS / YuqueFS / NotionFS: no official copy API; raises NotImplementedError.

Args:
    path1 (str): Source path (file or directory).
    path2 (str): Destination path.
    recursive (bool): If source is directory, whether to copy recursively; default False.
    **kwargs: Passed through (currently unused).
''')

_add_fs_chinese('LazyLLMFSBase.move', '''\
将文件或目录移动到目标路径。各子类调用对应平台的官方接口实现，基类默认抛出 NotImplementedError。

各平台支持情况：

- ObsidianFS：使用 shutil.move，目录移动需 recursive=True。
- S3FS：服务端 copy_object + delete_object，目录需 recursive=True。
- GoogleDriveFS：使用 files.update 的 addParents/removeParents（文件和文件夹均支持）。
- OneDriveFS：使用 Graph API PATCH /items/{id} 更新 parentReference。
- FeishuFS：使用 Drive v1 /move 接口（文件和文件夹均支持，文件夹异步执行）。
- FeishuWikiFS：使用 Wiki v2 /move 接口，支持节点（含子节点）的移动。
- ConfluenceFS：使用 REST API PUT /content/{id}/move 接口。
- OnesFS：使用 /pages/{id}/update 接口更新 parent_uuid。
- YuqueFS / NotionFS：官方接口不支持，抛出 NotImplementedError。

Args:
    path1 (str): 源路径（文件或目录）。
    path2 (str): 目标路径。
    recursive (bool): 源为目录时是否递归移动，默认 False。
    **kwargs: 透传（当前未使用）。
''')
_add_fs_english('LazyLLMFSBase.move', '''\
Move file or directory to the target path. Each subclass calls the provider's official API; the base class raises NotImplementedError.

Platform support:

- ObsidianFS: uses shutil.move; directory move requires recursive=True.
- S3FS: server-side copy_object + delete_object; directory move requires recursive=True.
- GoogleDriveFS: uses files.update with addParents/removeParents (supports files and folders).
- OneDriveFS: uses Graph API PATCH /items/{id} to update parentReference.
- FeishuFS: uses Drive v1 /move API (supports files and folders; folders are async).
- FeishuWikiFS: uses Wiki v2 /move API (supports nodes including children).
- ConfluenceFS: uses REST API PUT /content/{id}/move endpoint.
- OnesFS: uses /pages/{id}/update endpoint to change parent_uuid.
- YuqueFS / NotionFS: no official move API; raises NotImplementedError.

Args:
    path1 (str): Source path (file or directory).
    path2 (str): Destination path.
    recursive (bool): If source is directory, whether to move recursively; default False.
    **kwargs: Passed through (currently unused).
''')

_add_fs_chinese('LazyLLMFSBase.put_file', '''\
将本地文件上传到远程路径。

Args:
    lpath (str): 本地文件路径。
    rpath (str): 远程路径。
    **kwargs: 透传。
''')
_add_fs_english('LazyLLMFSBase.put_file', '''\
Upload local file to remote path.

Args:
    lpath (str): Local file path.
    rpath (str): Remote path.
    **kwargs: Passed through.
''')

_add_fs_chinese('LazyLLMFSBase.get_file', '''\
将远程文件下载到本地路径。

Args:
    rpath (str): 远程文件路径。
    lpath (str): 本地保存路径。
    **kwargs: 透传。
''')
_add_fs_english('LazyLLMFSBase.get_file', '''\
Download remote file to local path.

Args:
    rpath (str): Remote file path.
    lpath (str): Local path to save.
    **kwargs: Passed through.
''')

_add_fs_chinese('LazyLLMFSBase.materialize_dir', '''\
将远程目录递归物化到本地目录，并返回下载结果摘要。

Args:
    path (str): 远程目录路径。
    local_dir (str): 本地目标目录。
    **kwargs: 透传。

Returns:
    Dict[str, Any]: 包含 source_path、local_dir、materialized、file_count 和 files 的结果摘要。
''')
_add_fs_english('LazyLLMFSBase.materialize_dir', '''\
Recursively materialize a remote directory into a local directory and return a download summary.

Args:
    path (str): Remote directory path.
    local_dir (str): Local target directory.
    **kwargs: Passed through.

Returns:
    Dict[str, Any]: Summary containing source_path, local_dir, materialized, file_count, and files.
''')

_add_fs_chinese('LazyLLMFSBase.close', '''\
关闭 FS 占用的资源（如 HTTP session）。基类实现为关闭 _session；子类若持有其他资源可覆盖并在最后调用 super().close()。调用 close 后不应再使用该 FS 实例。
''')
_add_fs_english('LazyLLMFSBase.close', '''\
Release resources held by the FS (e.g. HTTP session). Base implementation closes _session; subclasses may override and call super().close() after releasing their own resources. The FS instance should not be used after close().
''')

_add_fs_chinese('LazyLLMFSBase.supports_webhook', '''\
返回当前 FS 是否支持 webhook。基类默认 False；支持 webhook 的子类覆盖 _platform_supports_webhook 返回 True。

Returns:
    bool: 是否支持 webhook。
''')
_add_fs_english('LazyLLMFSBase.supports_webhook', '''\
Return whether this FS supports webhooks. Base default is False; subclasses override _platform_supports_webhook to return True.

Returns:
    bool: True if webhook is supported.
''')

_add_fs_chinese('LazyLLMFSBase.register_webhook', '''\
在指定 path 注册 webhook。若 supports_webhook() 为 False 则返回 {'mode': 'none'}；否则调用 _register_webhook。
基类提供 _register_webhook 的默认实现（返回 {'mode': 'none'}）；支持 webhook 的子类应覆盖 _platform_supports_webhook 为 True 并覆盖 _register_webhook(webhook_url, events, path) 完成实际注册。

Args:
    path (str): 监控路径（如桶路径、目录 ID 等，由子类约定）。
    webhook_url (str): 回调 URL。
    events (list[str], optional): 事件类型列表，默认 ['*']。

Returns:
    dict: 含 mode（webhook/none）及子类返回的注册信息。
''')
_add_fs_english('LazyLLMFSBase.register_webhook', '''\
Register a webhook for the given path. Returns {'mode': 'none'} if supports_webhook() is False; otherwise calls _register_webhook.
Base class provides a default _register_webhook that returns {'mode': 'none'}; subclasses that support webhook override _platform_supports_webhook to return True and override _register_webhook(webhook_url, events, path) to perform registration.

Args:
    path (str): Path to watch (e.g. bucket or dir id; semantics defined by subclass).
    webhook_url (str): Callback URL.
    events (list[str], optional): Event types; default ['*'].

Returns:
    dict: mode (webhook/none) and subclass-specific registration info.
''')

_add_fs_chinese('LazyLLMFSBase._register_webhook', '''\
Webhook 注册钩子：子类可覆盖。基类默认实现返回 {'mode': 'none'}。
当 _platform_supports_webhook 为 True 时，register_webhook 会调用本方法；支持 webhook 的子类应覆盖本方法，执行平台侧注册并返回至少含 'mode' 的 dict（如 'webhook'）。

Args:
    webhook_url (str): 回调 URL。
    events (list[str]): 事件类型列表。
    path (str): 监控路径。
''')
_add_fs_english('LazyLLMFSBase._register_webhook', '''\
Webhook registration hook; subclasses may override. Default implementation returns {'mode': 'none'}.
When _platform_supports_webhook is True, register_webhook calls this method; subclasses that support webhook should override it to perform platform registration and return a dict with at least 'mode' (e.g. 'webhook').

Args:
    webhook_url (str): Callback URL.
    events (list[str]): Event types.
    path (str): Path to watch.
''')

_add_fs_chinese('LazyLLMFSBase.ensure_token', '''\
Token 刷新钩子：基类自动管理。
每次 inject_auth_header / _request 发请求前会调用本方法；当 access token 即将过期或未初始化时，会通过 _do_refresh_token / _do_acquire_without_refresh / _do_oauth_flow 重新获取。
''')
_add_fs_english('LazyLLMFSBase.ensure_token', '''\
Token refresh hook; managed by the base class.
Called before each inject_auth_header / _request; refreshes the token via _do_refresh_token / _do_acquire_without_refresh / _do_oauth_flow when the access token is missing or about to expire.
''')

_add_fs_chinese('LazyLLMFSBase._setup_auth', '''\
子类必须实现：在 __init__ 中调用，用于设置请求认证（如 session headers、Bearer token）。
''')
_add_fs_english('LazyLLMFSBase._setup_auth', '''\
Must be implemented by subclass: called from __init__ to set up request auth (e.g. session headers, Bearer token).
''')

_add_fs_chinese('LazyLLMFSBase._open', '''\
子类必须实现：打开 path 对应文件，返回 CloudFSBufferedFile 实例，供 fsspec 风格 open() 使用。

Args:
    path (str): 文件路径。
    mode (str): 如 'rb'、'wb'。
    block_size (int, optional): 缓冲块大小。
    autocommit (bool): 是否自动提交写入。
    cache_options (dict, optional): 缓存选项。
    **kwargs: 扩展参数。

Returns:
    CloudFSBufferedFile: 支持 _fetch_range（读）与 _upload_chunk（写）的缓冲文件。
''')
_add_fs_english('LazyLLMFSBase._open', '''\
Must be implemented by subclass: open file at path and return CloudFSBufferedFile for fsspec-style open().

Args:
    path (str): File path.
    mode (str): e.g. 'rb', 'wb'.
    block_size (int, optional): Buffer block size.
    autocommit (bool): Whether to autocommit on write.
    cache_options (dict, optional): Cache options.
    **kwargs: Extra options.

Returns:
    CloudFSBufferedFile: Buffered file with _fetch_range (read) and _upload_chunk (write).
''')

_add_fs_chinese('LazyLLMFSBase._download_range', '''\
子类必须实现：按字节区间 [start, end) 下载 path 对应内容，供 CloudFSBufferedFile 读使用。

Args:
    path (str): 文件路径。
    start (int): 起始字节（含）。
    end (int): 结束字节（不含）。

Returns:
    bytes: 该区间的数据。
''')
_add_fs_english('LazyLLMFSBase._download_range', '''\
Must be implemented by subclass: download bytes [start, end) for path; used by CloudFSBufferedFile for reading.

Args:
    path (str): File path.
    start (int): Start byte (inclusive).
    end (int): End byte (exclusive).

Returns:
    bytes: Data in that range.
''')

_add_fs_chinese('LazyLLMFSBase._upload_data', '''\
子类必须实现：将 data 完整上传到 path，供 CloudFSBufferedFile 写与 put_file 使用。

Args:
    path (str): 远程文件路径。
    data (bytes): 要上传的完整内容。
''')
_add_fs_english('LazyLLMFSBase._upload_data', '''\
Must be implemented by subclass: upload full data to path; used by CloudFSBufferedFile write and put_file.

Args:
    path (str): Remote file path.
    data (bytes): Full content to upload.
''')

# LinkDocumentFSBase
_add_fs_chinese('LinkDocumentFSBase', '''\
URL 文档文件系统基类，面向飞书、Notion、Obsidian 等可通过链接定位文档的系统。它在 LazyLLMFSBase 的基础上统一沉淀“解析链接、读取正文并附带引用、获取文档 ID、列出可编辑块、更新文本块”等文档流 API。

子类只需要实现平台差异：如何解析浏览器链接、如何读取正文、如何列出引用/块、如何更新块。若某个平台不支持块级编辑，应在对应方法中抛出 NotImplementedError。

Args:
    token (Any): 平台访问凭证；动态鉴权时可为空，由 dynamic_fs_auth 注入。
    base_url (str, optional): 平台 API 根地址。
    dynamic_auth (bool): 是否启用动态凭证。启用后会从 lazyllm.globals.config["dynamic_fs_auth"] 中按协议名读取 token。
''')
_add_fs_english('LinkDocumentFSBase', '''\
Base filesystem for URL-addressable document systems such as Feishu, Notion, and Obsidian. It extends LazyLLMFSBase with a shared document workflow: resolving browser links, reading content with references, getting provider document ids, listing editable blocks, and updating text blocks.

Subclasses keep only provider-specific behavior: how to resolve browser links, read content, collect references/blocks, and update blocks. Providers without block-level editing should raise NotImplementedError for the editing methods.

Args:
    token (Any): Provider access credential; may be empty when dynamic_auth=True.
    base_url (str, optional): Provider API base URL.
    dynamic_auth (bool): Whether to use dynamic credentials from lazyllm.globals.config["dynamic_fs_auth"] by protocol name.
''')
_add_fs_example('LinkDocumentFSBase', '''\
>>> from lazyllm.tools.fs import FeishuFS, NotionFS
>>> notion = NotionFS(token='ntn_xxx')
>>> meta = notion.resolve_link('https://www.notion.so/0123456789abcdef0123456789abcdef')
>>> content = notion.read_with_references(meta['path'])
>>> feishu = FeishuFS(app_id='cli_xxx', app_secret='xxx', space_id='dynamic')
>>> feishu.resolve_link('https://example.feishu.cn/wiki/xxxx')
''')

_add_fs_chinese('LinkDocumentFSBase.build_public_apis', '''\
构造面向工具暴露的公共 API 列表。文档型 FS 可复用 LazyLLMFSBase 的基础文件 API 与
LinkDocumentFSBase 的文档 API，并通过 extra / exclude 对平台差异做少量增删。

Args:
    extra (List[str], optional): 需要追加暴露的平台专属方法名。
    exclude (List[str], optional): 需要从默认集合中排除的方法名。

Returns:
    List[str]: 去重且保持声明顺序的公共 API 名称列表。
''')
_add_fs_english('LinkDocumentFSBase.build_public_apis', '''\
Build the public API list exposed to tools. Document filesystems can reuse LazyLLMFSBase file APIs
and LinkDocumentFSBase document APIs, then use extra / exclude for provider-specific differences.

Args:
    extra (List[str], optional): Provider-specific method names to expose.
    exclude (List[str], optional): Method names to remove from the default set.

Returns:
    List[str]: Deduplicated public API names preserving declaration order.
''')

_add_fs_chinese('LinkDocumentFSBase.to_link_path', '''\
将浏览器链接编码成统一的内部 link path。算法侧可以把用户粘贴的 URL 先转成该路径，再交给普通 FS read/ls 流程处理。

Args:
    url (str): 原始浏览器 URL。

Returns:
    str: 形如 /~link/<encoded-url> 的内部路径。
''')
_add_fs_english('LinkDocumentFSBase.to_link_path', '''\
Encode a browser URL into the shared internal link path format. Algorithm code may normalize pasted URLs this way before using ordinary FS read/ls flows.

Args:
    url (str): Original browser URL.

Returns:
    str: Internal path like /~link/<encoded-url>.
''')

_add_fs_chinese('LinkDocumentFSBase.is_link_path', '''\
判断路径是否为 LinkDocumentFSBase 生成的内部 link path。用于区分普通平台路径和用户直接粘贴的外部链接。

Args:
    path (str): 待判断路径。

Returns:
    bool: 是内部 link path 时返回 True。
''')
_add_fs_english('LinkDocumentFSBase.is_link_path', '''\
Return whether a path is an internal link path generated by LinkDocumentFSBase. This distinguishes provider paths from user-pasted external links.

Args:
    path (str): Path to inspect.

Returns:
    bool: True if the path is an internal link path.
''')

_add_fs_chinese('LinkDocumentFSBase.decode_link_path', '''\
从内部 link path 还原原始浏览器 URL。若 path 不是 link path，会抛出 ValueError，避免把普通平台路径误当作 URL。

Args:
    path (str): 形如 /~link/<encoded-url> 的内部路径。

Returns:
    str: 解码后的原始 URL。
''')
_add_fs_english('LinkDocumentFSBase.decode_link_path', '''\
Decode an internal link path back to the original browser URL. Raises ValueError when the input is not a link path, preventing provider paths from being treated as URLs.

Args:
    path (str): Internal path like /~link/<encoded-url>.

Returns:
    str: Decoded original URL.
''')

_add_fs_chinese('LinkDocumentFSBase.dedupe_document_references', '''\
按 url 字段对文档引用去重，并保持首次出现的顺序。用于统一飞书、Notion 等平台的引用收集结果，避免回答末尾重复列出同一链接。

Args:
    refs (List[Dict[str, Any]]): 引用对象列表，每项通常包含 url、ref_type 等字段。

Returns:
    List[Dict[str, Any]]: 去重后的引用列表。
''')
_add_fs_english('LinkDocumentFSBase.dedupe_document_references', '''\
Deduplicate document references by the url field while preserving first-seen order. This normalizes reference lists from providers such as Feishu and Notion before appending them to answers.

Args:
    refs (List[Dict[str, Any]]): Reference objects, typically containing url and ref_type.

Returns:
    List[Dict[str, Any]]: Deduplicated references.
''')

_add_fs_chinese('LinkDocumentFSBase.format_document_references_footer', '''\
将引用列表格式化成可追踪的文本页脚。页脚使用 provider 名称作为边界标识，便于上层工具在回答或日志中保留来源链接。

Args:
    refs (List[Dict[str, Any]]): 引用对象列表。
    provider (str): 平台名，例如 feishu 或 notion。

Returns:
    str: 格式化后的引用页脚；无引用时返回空字符串。
''')
_add_fs_english('LinkDocumentFSBase.format_document_references_footer', '''\
Format references into a traceable text footer. The footer uses the provider name in its boundary marker so upstream tools can keep source links in answers or logs.

Args:
    refs (List[Dict[str, Any]]): Reference objects.
    provider (str): Provider name, such as feishu or notion.

Returns:
    str: Formatted references footer, or an empty string when refs is empty.
''')

_add_fs_chinese('LinkDocumentFSBase.resolve_link', f'''\
解析浏览器链接、provider URI 或平台路径，返回标准化文档元信息。建议在用户粘贴链接后先调用本方法，再读取正文或继续展开子页面。

{_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    url_or_path (str): 浏览器 URL、provider URI 或平台路径。

Returns:
    Dict[str, Any]: 标准化元信息，至少尽量包含 provider、object_id、object_type、title、path、has_child 等字段。
''')
_add_fs_english('LinkDocumentFSBase.resolve_link', f'''\
Resolve a browser URL, provider URI, or provider path into normalized document metadata. Call this before reading content or expanding child pages when users paste a document link.

{_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    url_or_path (str): Browser URL, provider URI, or provider-specific path.

Returns:
    Dict[str, Any]: Normalized metadata, preferably including provider, object_id, object_type, title, path, and has_child.
''')

_add_fs_chinese('LinkDocumentFSBase.read_with_references', f'''\
读取文档正文，并在平台支持时追加可追踪的引用页脚。适合总结、分析、跨页面阅读等需要保留链接/关系信息的场景。

{_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    path (str): 浏览器 URL、provider URI 或平台路径。

Returns:
    str: UTF-8 文档正文；若平台返回引用，则末尾追加 references footer。
''')
_add_fs_english('LinkDocumentFSBase.read_with_references', f'''\
Read document content and append a traceable references footer when the provider supports references. Useful for summarization, analysis, and linked-page context.

{_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    path (str): Browser URL, provider URI, or provider-specific path.

Returns:
    str: UTF-8 document content, optionally followed by a references footer.
''')

_add_fs_chinese('LinkDocumentFSBase.fetch_url', '''\
通过浏览器 URL 直接读取文档原始字节内容。基类默认转调 read_bytes(url)，子类可覆盖以适配平台 URL 解析。

Args:
    url (str): 平台浏览器 URL。

Returns:
    bytes: 文档原始字节内容。
''')
_add_fs_english('LinkDocumentFSBase.fetch_url', '''\
Fetch raw document bytes from a browser URL. The base implementation delegates to read_bytes(url); providers may override it for platform-specific URL parsing.

Args:
    url (str): Provider browser URL.

Returns:
    bytes: Raw document content bytes.
''')

_add_fs_chinese('LinkDocumentFSBase.get_document_id', '''\
获取文档的 provider-native ID。该 ID 可用于块级编辑、审计日志或后端绑定。

Args:
    path (str): 浏览器 URL、provider URI 或平台路径。

Returns:
    str: 平台原生文档 ID。
''')
_add_fs_english('LinkDocumentFSBase.get_document_id', '''\
Return the provider-native document id. The id can be used for block-level editing, audit logs, or backend bindings.

Args:
    path (str): Browser URL, provider URI, or provider-specific path.

Returns:
    str: Provider-native document id.
''')

_add_fs_chinese('LinkDocumentFSBase.get_doc_blocks', '''\
列出文档中的可编辑块。返回值用于精确修改文本块，避免整篇文档覆盖导致格式丢失。

Args:
    path (str): 浏览器 URL、provider URI 或平台路径。
    with_descendants (bool): 是否包含嵌套子块，默认 True。

Returns:
    List[Dict[str, Any]]: 块列表，通常包含 block_id、block_type、parent_id、plain_text 等字段。
''')
_add_fs_english('LinkDocumentFSBase.get_doc_blocks', '''\
List editable blocks in a document. Use the returned blocks for precise text updates without replacing the whole document and losing formatting.

Args:
    path (str): Browser URL, provider URI, or provider-specific path.
    with_descendants (bool): Whether nested child blocks should be included. Defaults to True.

Returns:
    List[Dict[str, Any]]: Blocks, typically including block_id, block_type, parent_id, and plain_text.
''')

_add_fs_chinese('LinkDocumentFSBase.update_doc_block_text', '''\
更新单个文本块内容。该方法只应修改指定 block_id 对应的文本块；表格、图片等非文本块由子类按平台能力处理或拒绝。

Args:
    path (str): 文档浏览器 URL、provider URI 或平台路径。
    block_id (str): 来自 get_doc_blocks 的块 ID。
    new_text (str): 新文本内容。
''')
_add_fs_english('LinkDocumentFSBase.update_doc_block_text', '''\
Update one text block. The method should only modify the block identified by block_id; non-text blocks such as tables or images are provider-specific and may be rejected.

Args:
    path (str): Document browser URL, provider URI, or provider-specific path.
    block_id (str): Block id returned by get_doc_blocks.
    new_text (str): Replacement text.
''')

_add_fs_chinese('CloudFsWatchdog', '''\
云文件系统监听辅助类。基于 LazyLLMFSBase 提供的 ls(detail=True) 做目录快照与差异比较，
并在变更时触发用户注册的回调；若底层 FS 支持 webhook，则可通过 register_webhook 结合平台回调使用。

FS 子类只负责各云平台 API 差异与 webhook 实际注册逻辑；CloudFsWatchdog 仅负责通用的轮询线程与事件分发。
''')
_add_fs_english('CloudFsWatchdog', '''\
Helper for watching cloud filesystems. It polls LazyLLMFSBase.ls(detail=True) to compute snapshots,
diffs changes, and dispatches events to user handlers; when the underlying FS supports webhooks,
it can delegate registration via register_webhook.

FS subclasses handle platform-specific APIs and webhook registration; CloudFsWatchdog only manages
generic polling thread and event dispatch.
''')

_add_fs_chinese('CloudFsWatchdog.watch', '''\
对 path 进行轮询监控，变更时调用 callback。返回 watcher_id 便于后续 unwatch。

Args:
    path (str): 要监控的目录路径。
    callback (Callable): (event, name, info) -> None，event 为 'created'/'deleted'/'modified'。
    polling_interval (int): 轮询间隔（秒），默认 30。

Returns:
    str: watcher_id。
''')
_add_fs_english('CloudFsWatchdog.watch', '''\
Poll path for changes and invoke callback. Returns watcher_id for unwatch.

Args:
    path (str): Directory path to watch.
    callback (Callable): (event, name, info) -> None; event is 'created'/'deleted'/'modified'.
    polling_interval (int): Poll interval in seconds; default 30.

Returns:
    str: watcher_id.
''')

_add_fs_chinese('CloudFsWatchdog.on', '''\
注册对特定事件集合的处理函数，相当于 watch 的语法糖。

Args:
    path (str): 监控路径。
    events (list[str]): 事件类型列表，例如 ['created', 'deleted']。
    handler (Callable[[str, str, dict], None]): 处理函数。
    polling_interval (int): 轮询间隔秒数。
''')
_add_fs_english('CloudFsWatchdog.on', '''\
Register handler for a set of events; sugar on top of watch.

Args:
    path (str): Path to watch.
    events (list[str]): Events such as ['created', 'deleted'].
    handler (Callable[[str, str, dict], None]): Handler.
    polling_interval (int): Polling interval in seconds.
''')

_add_fs_chinese('CloudFsWatchdog.register_webhook', '''\
若底层 FS 支持 webhook，则调用其 register_webhook；否则回退到轮询模式或返回 mode 'none'。

Args:
    path (str): 监控路径。
    webhook_url (str): 回调 URL。
    events (list[str], optional): 事件类型，默认 ['*']。
    callback (Callable, optional): 若需同时轮询回调可传入。

Returns:
    dict: 含 mode（webhook/polling/none）及平台返回信息或 watcher_id。
''')
_add_fs_english('CloudFsWatchdog.register_webhook', '''\
Call underlying FS.register_webhook if supported; otherwise fall back to polling or mode 'none'.

Args:
    path (str): Path to watch.
    webhook_url (str): Callback URL.
    events (list[str], optional): Event types; default ['*'].
    callback (Callable, optional): Optional callback for polling.

Returns:
    dict: mode (webhook/polling/none) and platform info or watcher_id.
''')

_add_fs_chinese('CloudFsWatchdog.unwatch', '''\
取消指定 watcher。

Args:
    watcher_id (str): watch 返回的 id。

Returns:
    bool: 是否成功移除。
''')
_add_fs_english('CloudFsWatchdog.unwatch', '''\
Remove a watcher by id.

Args:
    watcher_id (str): Id returned by watch.

Returns:
    bool: True if removed.
''')

_add_fs_chinese('CloudFsWatchdog.stop', '''\
请求轮询线程在当前睡眠周期结束后停止。调用后线程会在下一轮循环退出，无需传参。
''')
_add_fs_english('CloudFsWatchdog.stop', '''\
Request the polling thread to stop after the current sleep cycle. No arguments; thread exits on next loop.
''')

_add_fs_chinese('CloudFsWatchdog.supports_webhook', '''\
返回底层 FS 是否支持 webhook。若子类实现 supports_webhook 或 _platform_supports_webhook 则据此判断，否则为 False。

Returns:
    bool: 是否支持 webhook。
''')
_add_fs_english('CloudFsWatchdog.supports_webhook', '''\
Return whether the underlying FS supports webhooks. Uses subclass supports_webhook or _platform_supports_webhook if present; else False.

Returns:
    bool: True if webhook is supported.
''')

_add_fs_chinese('CloudFsWatchdog.on_created', '''\
仅监听 created 事件，等价于 on(path, ['created'], handler, ...)。

Args:
    path (str): 监控路径。
    handler (Callable): (event, name, info) -> None。
    polling_interval (int): 轮询间隔（秒），默认 30。

Returns:
    str: watcher_id。
''')
_add_fs_english('CloudFsWatchdog.on_created', '''\
Watch only for created events; same as on(path, ['created'], handler, ...).

Args:
    path (str): Path to watch.
    handler (Callable): (event, name, info) -> None.
    polling_interval (int): Poll interval in seconds; default 30.

Returns:
    str: watcher_id.
''')

_add_fs_chinese('CloudFsWatchdog.on_deleted', '''\
仅监听 deleted 事件，等价于 on(path, ['deleted'], handler, ...)。

Args:
    path (str): 监控路径。
    handler (Callable): (event, name, info) -> None。
    polling_interval (int): 轮询间隔（秒），默认 30。

Returns:
    str: watcher_id。
''')
_add_fs_english('CloudFsWatchdog.on_deleted', '''\
Watch only for deleted events; same as on(path, ['deleted'], handler, ...).

Args:
    path (str): Path to watch.
    handler (Callable): (event, name, info) -> None.
    polling_interval (int): Poll interval in seconds; default 30.

Returns:
    str: watcher_id.
''')

_add_fs_chinese('CloudFsWatchdog.on_modified', '''\
仅监听 modified 事件，等价于 on(path, ['modified'], handler, ...)。

Args:
    path (str): 监控路径。
    handler (Callable): (event, name, info) -> None。
    polling_interval (int): 轮询间隔（秒），默认 30。

Returns:
    str: watcher_id。
''')
_add_fs_english('CloudFsWatchdog.on_modified', '''\
Watch only for modified events; same as on(path, ['modified'], handler, ...).

Args:
    path (str): Path to watch.
    handler (Callable): (event, name, info) -> None.
    polling_interval (int): Poll interval in seconds; default 30.

Returns:
    str: watcher_id.
''')

_add_fs_example('CloudFsWatchdog', '''\
>>> from lazyllm.tools.fs import CloudFS, CloudFsWatchdog
>>> fs = CloudFS(platform='s3', access_key='xxx', secret_key='yyy')
>>> watchdog = CloudFsWatchdog(fs)
>>> def on_change(event, name, info):
...     print(event, name, info)
>>> wid = watchdog.watch('/my-bucket', on_change, polling_interval=30)
>>> watchdog.on_created('/my-bucket', lambda e, n, i: print('new:', n))
>>> watchdog.unwatch(wid)
>>> watchdog.stop()
''')

# CloudFSBufferedFile
_add_fs_chinese('CloudFSBufferedFile', '''\
基于 fsspec.AbstractBufferedFile 的缓冲文件实现，用于云存储的区间下载与整块上传。
''')
_add_fs_english('CloudFSBufferedFile', '''\
Buffered file implementation based on fsspec.AbstractBufferedFile for range download and chunk upload on cloud storage.
''')


# FeishuFSBase (lazyllm.tools.fs.supplier.feishu)
_add_feishu_chinese('FeishuFSBase', '''\
飞书文件系统基类，供 FeishuFS、FeishuWikiFS 继承。使用飞书开放平台 API，支持应用凭证与用户 OAuth refresh_token。

Args:
    base_url (str, optional): 飞书开放平台 API 根地址，默认官方地址。
    app_id (str, optional): 企业自建应用 App ID；未传时从 config feishu_app_id 读取。
    app_secret (str, optional): 企业自建应用 App Secret；未传时从 config feishu_app_secret 读取。
    space_id (str, optional): 知识库空间 ID，子类 FeishuWikiFS 使用。
    user_refresh_token (str, optional): 用户 OAuth refresh_token；'auto' 表示启动时触发浏览器授权。
    oauth_port (int): OAuth 回调本地端口，默认 9981。
    oauth_scope (str): OAuth scope 字符串，默认含 offline_access 与常用 drive/wiki 权限。
    asynchronous (bool): 是否异步模式，默认 False。
    use_listings_cache (bool): 是否缓存目录列表，默认 False。
    skip_instance_cache (bool): 是否跳过实例缓存，默认 False。
    loop (Any, optional): 异步事件循环。
''')
_add_feishu_english('FeishuFSBase', '''\
Base class for Feishu filesystem implementations (FeishuFS, FeishuWikiFS). Uses Feishu Open Platform API; supports app credentials and user OAuth refresh_token.

Args:
    base_url (str, optional): Feishu Open Platform API base URL; default official endpoint.
    app_id (str, optional): Enterprise app App ID; read from config feishu_app_id when not passed.
    app_secret (str, optional): Enterprise app App Secret; read from config feishu_app_secret when not passed.
    space_id (str, optional): Wiki space ID, used by FeishuWikiFS.
    user_refresh_token (str, optional): User OAuth refresh_token; 'auto' triggers browser auth on init.
    oauth_port (int): Local port for OAuth callback, default 9981.
    oauth_scope (str): OAuth scope string; default includes offline_access and common drive/wiki scopes.
    asynchronous (bool): Whether async mode, default False.
    use_listings_cache (bool): Whether to cache directory listings, default False.
    skip_instance_cache (bool): Whether to skip instance cache, default False.
    loop (Any, optional): Async event loop.
''')
_add_feishu_chinese('FeishuFSBase.get_user_refresh_token', '''\
返回当前保存的用户 OAuth refresh_token（刷新后可能已更新为最新值）。供调用方在 OAuth 或刷新后持久化，下次构造时传入以跳过授权。

Returns:
    str: 当前 refresh_token，未设置或未授权时为空字符串。
''')
_add_feishu_english('FeishuFSBase.get_user_refresh_token', '''\
Return the current user OAuth refresh_token (may have been updated after refresh). For caller to persist after OAuth or refresh and pass to constructor on next run to skip auth.

Returns:
    str: Current refresh_token, or empty string if not set or not yet authorized.
''')
_add_feishu_chinese('FeishuFSBase.create_block', '''\
在指定飞书 Docx 文档的父块下创建原生块子树。

该方法是飞书创建嵌套块接口的薄封装，不解析插入位置，不将扁平 blocks 转换为 descendants，也不读取目标文档。调用方必须提供符合飞书原生请求格式的参数。

Args:
    document_id (str): 目标飞书 Docx 文档 ID。
    parent_block_id (str): 目标父块 ID。
    index (int): 在父块直接子块中的插入位置。
    children_id (List[str]): 本次创建的顶层临时块 ID。
    descendants (List[Dict[str, Any]]): 飞书原生 descendants。

Returns:
    Dict[str, Any]: 飞书响应中的原始 data。
''')
_add_feishu_english('FeishuFSBase.create_block', '''\
Create native descendant blocks under a parent in a Feishu Docx document.

This is a thin wrapper over Feishu's descendant-block endpoint. It does not resolve the insertion position,
convert flat blocks into descendants, or read the target document. The caller must supply native Feishu request values.

Args:
    document_id (str): Target Feishu Docx document ID.
    parent_block_id (str): Target parent block ID.
    index (int): Insertion position among the parent's direct children.
    children_id (List[str]): Temporary IDs of the top-level blocks to create.
    descendants (List[Dict[str, Any]]): Native Feishu descendants.

Returns:
    Dict[str, Any]: Raw data from the Feishu response.
''')
_add_feishu_chinese('FeishuFSBase.update_block', '''\
使用飞书原生 batch_update requests 批量更新文档块。

该方法不从完整 Block 推断更新操作，也不自动拆分超过飞书单次限制的请求。调用方负责生成、分批 requests 并串联文档版本。

Args:
    document_id (str): 目标飞书 Docx 文档 ID。
    requests (List[Dict[str, Any]]): 飞书原生 batch_update requests。
    document_revision_id (int): 要操作的文档版本，默认 -1。

Returns:
    Dict[str, Any]: 飞书响应中的原始 data。
''')
_add_feishu_english('FeishuFSBase.update_block', '''\
Update document blocks with native Feishu batch_update requests.

The method does not infer update operations from complete blocks or split requests that exceed Feishu's per-call limit.
The caller is responsible for generating and batching requests and chaining document revisions.

Args:
    document_id (str): Target Feishu Docx document ID.
    requests (List[Dict[str, Any]]): Native Feishu batch_update requests.
    document_revision_id (int): Document revision to update; defaults to -1.

Returns:
    Dict[str, Any]: Raw data from the Feishu response.
''')
_add_feishu_chinese('FeishuFSBase.delete_block', '''\
删除飞书父块下指定位置区间的直接子块。

该方法直接封装飞书 batch_delete 接口。它不接收目标 block_id，不读取文档，也不解析父块或同级位置。

Args:
    document_id (str): 目标飞书 Docx 文档 ID。
    parent_block_id (str): 目标父块 ID。
    start_index (int): 删除区间起始位置，包含。
    end_index (int): 删除区间结束位置，不包含。
    document_revision_id (int): 要操作的文档版本，默认 -1。

Returns:
    Dict[str, Any]: 飞书响应中的原始 data。

''')
_add_feishu_english('FeishuFSBase.delete_block', '''\
Delete a positional range of direct children under a Feishu parent block.

This method directly wraps Feishu's batch_delete endpoint. It does not accept a target block ID, read the document,
or resolve the parent or sibling position.

Args:
    document_id (str): Target Feishu Docx document ID.
    parent_block_id (str): Target parent block ID.
    start_index (int): Inclusive start of the child range.
    end_index (int): Exclusive end of the child range.
    document_revision_id (int): Document revision to update; defaults to -1.

Returns:
    Dict[str, Any]: Raw data from the Feishu response.

''')
_add_feishu_chinese('FeishuFSBase.move_block', '''\
通过创建后删除的方式移动一个已准备好的飞书 Block 子树。

飞书没有原生 Block Move 接口。该方法使用调用方提供的父块、位置和原生 descendants 先创建副本，
再使用创建返回的文档版本删除源块。target_index 表示移动完成后的最终位置；同父块移动时会自动修正创建和删除位置。
该方法不读取文档、定位块或生成 descendants。创建的 Block 使用新 ID，评论和块历史不会迁移，两次请求不是原子操作。
创建成功但删除失败时，目标副本会保留，并向调用方抛出删除异常。

Args:
    document_id (str): 目标飞书 Docx 文档 ID。
    source_parent_block_id (str): 源块的父块 ID。
    source_index (int): 源块在父块直接子块中的位置。
    target_parent_block_id (str): 目标父块 ID。
    target_index (int): 移动完成后在目标父块中的位置。
    children_id (List[str]): 包含单个顶层临时块 ID 的列表。
    descendants (List[Dict[str, Any]]): 飞书原生 descendants。

Returns:
    Dict[str, Any]: 包含 create 和 delete 两个飞书原始 data 的字典。

Raises:
    ValueError: children_id 不是单个子树根块。
''')
_add_feishu_english('FeishuFSBase.move_block', '''\
Move a prepared Feishu Block subtree by creating it and then deleting the source.

Feishu has no native Block Move endpoint. This method creates a copy from caller-supplied parents, positions,
and native descendants, then deletes the source using the revision returned by the create request. target_index is
the final position after the move; create and delete positions are adjusted for moves under the same parent. The
method does not read the document, locate blocks, or generate descendants. Created blocks receive new IDs; comments
and block history are not transferred, and the two requests are not atomic. If deletion fails after creation, the
destination copy remains and the deletion error is raised.

Args:
    document_id (str): Target Feishu Docx document ID.
    source_parent_block_id (str): Parent ID of the source block.
    source_index (int): Position of the source among its parent's direct children.
    target_parent_block_id (str): Target parent block ID.
    target_index (int): Final position under the target parent after the move.
    children_id (List[str]): A list containing the single temporary root block ID to create.
    descendants (List[Dict[str, Any]]): Native Feishu descendants.

Returns:
    Dict[str, Any]: A dictionary containing the raw create and delete response data.

Raises:
    ValueError: children_id does not contain exactly one subtree root.
''')
_add_feishu_chinese('FeishuFSBase.write_doc_blocks', '''\
向已有飞书 Docx 文档末尾追加原生块，并返回写入后重新读取的完整块列表。

输入是飞书原生块的扁平列表。每个块的 block_id 在本次写入中作为临时标识，父子关系仅由 parent_id 确定；输入中的 children 等读回元数据不会用于构建层级。源文档的 Page 块会被忽略，输入对象不会被修改。

该方法只负责写入指定文档，不负责通过 URL 或 Wiki 路径解析文档，不会创建或清空目标文档。调用方应先准备受飞书嵌套块创建接口支持的规范化块。

Args:
    document_id (str): 目标飞书 Docx 文档的 document_id，不能是浏览器 URL 或 Wiki 路径。
    blocks (List[Dict[str, Any]]): 要追加的飞书原生扁平块；使用 parent_id 表示层级。

Returns:
    List[Dict[str, Any]]: 写入完成后从目标文档重新读取的完整飞书原生块列表。

Raises:
    TypeError: blocks 不是列表。
    ValueError: blocks 为空，或包含当前结构化写入不支持的块类型。
''')
_add_feishu_english('FeishuFSBase.write_doc_blocks', '''\
Append native blocks to an existing Feishu Docx document and return the complete block list read back after writing.

The input is a flat list of native Feishu blocks. Each block_id is a temporary identifier for this write, and hierarchy is derived only from parent_id; read-side metadata such as children is not used to build the hierarchy. The source Page block is ignored, and the input objects are not mutated.

This method only writes to the specified document. It does not resolve browser URLs or Wiki paths, create the target document, or clear existing content. The caller must provide normalized blocks supported by Feishu's nested-block creation API.

Args:
    document_id (str): The target Feishu Docx document_id, not a browser URL or Wiki path.
    blocks (List[Dict[str, Any]]): Native flat Feishu blocks to append, with hierarchy expressed by parent_id.

Returns:
    List[Dict[str, Any]]: The complete native Feishu block list read from the target document after writing.

Raises:
    TypeError: blocks is not a list.
    ValueError: blocks is empty or contains a block type unsupported by structured writing.
''')
_add_feishu_chinese('FeishuWikiFile', '''\
飞书知识库节点对应的缓冲文件实现。打开时通过 FeishuWikiFS 拉取节点内容并缓存在内存，支持区间读。

Args:
    fs (FeishuWikiFS): 所属的飞书知识库文件系统实例。
    path (str): Wiki 节点路径。
    **kwargs: 透传给 CloudFSBufferedFile（如 mode、block_size 等）。
''')
_add_feishu_english('FeishuWikiFile', '''\
Buffered file for a Feishu wiki node. On open, fetches node content via FeishuWikiFS and caches in memory; supports range read.

Args:
    fs (FeishuWikiFS): The Feishu wiki filesystem instance.
    path (str): Wiki node path.
    **kwargs: Passed to CloudFSBufferedFile (e.g. mode, block_size).
''')

# FeishuFS
_add_fs_chinese('FeishuFS', f'''\
飞书云盘文件系统：使用飞书开放平台 Drive API，支持 ls、读写、mkdir、rm。目录监听需配合 CloudFsWatchdog；部分能力支持 webhook。

{_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH}

构造参数:
    base_url (str, optional): 飞书开放平台根地址，默认使用官方地址即可。
    app_id (str, optional): 企业自建应用的 App ID，用于换取 tenant_access_token 或刷新 user_access_token。
    app_secret (str, optional): 企业自建应用的 App Secret。
    space_id (str, optional): 若传入则会返回 FeishuWikiFS 实例，将指定知识库作为 FS。
        传入真实 space_id（如 'wikcnKQ1k3pcuo5uSK4t8VN6kVf'）或传入 'dynamic'（运行时从
        globals.config['feishu_wiki_space_id'] 或链接 get_node 响应中动态解析）。
    云盘上传：put_file 时若远程路径以 .md 结尾或传入 content_type='markdown'，会在目标目录创建 docx 并按 Markdown 解析写入（标题、列表、代码块等）；否则按二进制上传原文件。
    user_refresh_token (str, optional): OAuth2 refresh_token，用于以用户身份访问「我的空间」个人文件。
        - 传入真实 refresh_token：直接用于换取 user_access_token，每次刷新后内存中的 token 同步滚动。
        - 传入 'auto'：自动触发 OAuth2 授权流程——在本地启动临时回调服务，将授权链接通过
          lazyllm.LOG.success 输出到终端，用户在浏览器中点击并完成授权后，自动交换得到 refresh_token。
        飞书 refresh_token 有效期约 7 天，每次使用后立即作废并换发新值；只要在有效期内使用即可持续续期
        （最长 365 天后需重新 OAuth 授权）。token 的持久化由调用方负责，LazyLLM 不做本地存储。
        若不设置则使用 tenant_access_token（仅能访问已授权给应用的共享文件）。
    oauth_port (int, optional): 'auto' 流程中本地回调服务的监听端口，默认 9981。
        使用前需在飞书应用「安全设置」→「重定向 URL」中添加 http://localhost:{{oauth_port}}/callback。
    oauth_scope (str, optional): 'auto' 流程中请求的 OAuth2 scope，多个用空格分隔。
        默认值已包含 offline_access 及常用云盘权限，通常无需修改。

认证与配置:
    - 应用场景（共享文件/企业共享云盘）: 传入 app_id + app_secret，FS 自动获取 tenant_access_token。
    - 用户场景（「我的空间」个人文件）: 额外传入 user_refresh_token，FS 自动换取并续期 user_access_token。
配置与环境变量: feishu_app_id / FEISHU_APP_ID、feishu_app_secret / FEISHU_APP_SECRET；未传时从 config 读取。
    user_refresh_token 不支持环境变量，须通过构造函数显式传入。

⚠️  权限配置说明（user_refresh_token 模式的关键前提）:
    飞书区分「应用权限」（tenant_access_token 使用）和「用户权限」（user_access_token 使用），两者相互独立，
    必须分别配置。user_refresh_token 模式仅使用「用户权限」，即使同名的应用权限已开通也不生效。

    必须开通的「用户权限」（在开发者后台「权限管理」中，授权类型须包含「用户授权」）:
        个人云盘（drive:drive）:
        - offline_access              【必须】用于获取 refresh_token，没有此权限将无法拿到 refresh_token
        - drive:drive                 【推荐】读写个人云盘文件
        - drive:drive:readonly        【可选】只读个人云盘（与 drive:drive 二选一即可）
        - drive:drive.metadata:readonly  【可选】读取文件元数据

        知识库（wiki/wikis）:
        - wiki:wiki:readonly          【必须】列出知识库节点（ls）
        - docx:document:readonly      【必须】读取知识库内的 docx 文档内容（open）
        - wiki:wiki                   【可选】读写知识库（如只需只读可不开）

    注意: 在「权限管理」中，每条权限的「授权类型」列必须包含「用户授权」选项才对 OAuth 生效；
    若只有「应用授权」，则 user_access_token 无法获得该权限，OAuth 授权页面中也不会展示该权限。

如何配置:
    1. 打开飞书开放平台 https://open.feishu.cn，使用管理员账号登录。
    2. 进入「开发者后台」→「创建企业自建应用」，填写名称与描述并创建。
    3. 在应用详情「凭证与基础信息」中获取 App ID、App Secret，通过构造函数传入。
    4. 进入「权限管理」，搜索并开启以下权限，确认每条权限的授权类型包含「用户授权」:
         offline_access、drive:drive（或 drive:drive:readonly）、drive:drive.metadata:readonly
         wiki:wiki:readonly、docx:document:readonly
       完成后点击「版本管理与发布」→「创建并发布版本」使权限生效。
    5. 在「安全设置」→「重定向 URL」中添加 http://localhost:9981/callback（或对应 oauth_port）。
    6. 传入 user_refresh_token='auto' 即可触发授权流程；授权完成后可通过 get_user_refresh_token() 取出
       最新 token 由调用方持久化，下次直接传入该 token 跳过 OAuth 流程。
''')
_add_fs_english('FeishuFS', f'''\
Feishu (Lark) drive FS: uses Feishu Open Platform Drive API; supports ls, read/write, mkdir, rm. Use CloudFsWatchdog for watching; webhook supported where applicable.

{_FEISHU_DOCUMENT_LINK_WORKFLOW_EN}

Parameters:
    base_url (str, optional): Feishu Open Platform base URL; defaults to the official endpoint.
    app_id (str, optional): App ID of the enterprise custom app, used to obtain tenant_access_token or refresh user_access_token.
    app_secret (str, optional): App Secret of the enterprise custom app.
    space_id (str, optional): When provided, returns a FeishuWikiFS instance targeting the given wiki space.
        Pass a real space_id (e.g. 'wikcnKQ1k3pcuo5uSK4t8VN6kVf') or 'dynamic' to resolve the space
        at runtime from globals.config['feishu_wiki_space_id'] or from the get_node response.
    Drive upload: when the remote path ends with .md or put_file(..., content_type='markdown'), a docx is created in the target folder and content is parsed as Markdown; otherwise the file is uploaded as binary.
    user_refresh_token (str, optional): OAuth2 refresh_token for accessing the user's personal drive ("My Space").
        - Pass a real refresh_token: used directly to obtain user_access_token; rolls forward in memory after each use.
        - Pass 'auto': triggers the OAuth2 flow automatically — a local callback server is started, the authorization
          URL is printed via lazyllm.LOG.success, and the token is obtained once the user completes browser auth.
        Feishu refresh_tokens are valid for ~7 days and invalidated immediately on use; as long as the token is used
        within that window it renews indefinitely (up to 365 days before re-authorization is needed).
        Token persistence is the caller's responsibility; LazyLLM does not store it locally.
        Without this, tenant_access_token is used, which only accesses files shared with the app.
    oauth_port (int, optional): Local port for the OAuth2 callback server used in the 'auto' flow. Default: 9981.
        http://localhost:{{oauth_port}}/callback must be pre-registered in the Feishu app Security Settings → Redirect URL.
    oauth_scope (str, optional): OAuth2 scope string (space-separated) for the 'auto' flow.
        The default already includes offline_access and common drive scopes; override only when necessary.

Auth and config:
    - App scenario (shared/enterprise files): pass app_id + app_secret; FS obtains tenant_access_token automatically.
    - User scenario (personal "My Space" files): also pass user_refresh_token; FS exchanges and renews
      user_access_token automatically.
Config and env: feishu_app_id / FEISHU_APP_ID, feishu_app_secret / FEISHU_APP_SECRET; resolved from config when not passed.
    user_refresh_token does not support env var and must be passed explicitly to the constructor.

⚠️  Permission configuration (critical prerequisite for user_refresh_token mode):
    Feishu distinguishes between "application permissions" (used by tenant_access_token) and
    "user permissions" (used by user_access_token). They are configured independently.
    user_refresh_token mode uses ONLY user permissions; application permissions for the same scope
    have no effect on user_access_token.

    Required USER permissions (in Developer Console → Permission Management,
    the "Authorization Type" column must include "User Authorization"):
        Personal Drive (drive:drive):
        - offline_access              [REQUIRED] needed to obtain a refresh_token; without this no
                                      refresh_token is returned and user_refresh_token flow will not work
        - drive:drive                 [recommended] read/write personal drive files
        - drive:drive:readonly        [optional] read-only personal drive (either this or drive:drive)
        - drive:drive.metadata:readonly  [optional] read file metadata

        Wiki / Knowledge Base (FeishuWikiFS):
        - wiki:wiki:readonly          [REQUIRED] list wiki nodes (ls)
        - docx:document:readonly      [REQUIRED] read docx document content (open)
        - wiki:wiki                   [optional] read/write wiki (omit if read-only is sufficient)

    Note: for each permission in Permission Management, the "Authorization Type" field must include
    the "User Authorization" option for it to take effect in OAuth. Permissions that only show
    "Application Authorization" will not appear in the OAuth consent screen and will not be granted
    to user_access_token, even if they are enabled as application permissions.

How to configure:
    1. Go to https://open.feishu.cn, log in with an admin account.
    2. In Developer Console, create an enterprise custom app; get App ID and App Secret; pass them to the constructor.
    3. In Permission Management, search for and enable the following permissions, verifying that each one's
       Authorization Type includes "User Authorization":
         offline_access, drive:drive (or drive:drive:readonly), drive:drive.metadata:readonly,
         wiki:wiki:readonly, docx:document:readonly
       Then go to Version Management → Create and Publish Version to make the permissions take effect.
    4. In Security Settings → Redirect URL, add http://localhost:9981/callback (or the chosen oauth_port).
    5. Pass user_refresh_token='auto' to trigger the OAuth flow on construction; the authorization URL is printed to
       the terminal and the user completes auth in a browser. Afterwards call get_user_refresh_token() to retrieve
       the token for persistence; pass it directly on subsequent runs to skip the OAuth step.
''')
_add_fs_example('FeishuFS', '''\
>>> from lazyllm.tools.fs import FeishuFS, FS
>>> import lazyllm
>>> # tenant_access_token: access files shared with the app
>>> fs = FeishuFS(app_id='cli_xxx', app_secret='xxx')
>>> fs.ls('/')
>>> # user_access_token via auto OAuth flow (browser auth, one-time setup)
>>> fs_auto = FeishuFS(app_id='cli_xxx', app_secret='xxx', user_refresh_token='auto')
>>> # lazyllm.LOG.success prints the auth URL; user opens it and completes auth
>>> fs_auto.ls('/')
>>> refresh_token = fs_auto.get_user_refresh_token()  # persist for next run
>>> # subsequent runs: pass the saved token directly to skip OAuth
>>> fs_user = FeishuFS(app_id='cli_xxx', app_secret='xxx', user_refresh_token=refresh_token)
>>> fs_user.ls('/')
>>> # use FeishuWikiFS implicitly by passing space_id (wiki space id)
>>> wiki_fs = FeishuFS(app_id='cli_xxx', app_secret='xxx', space_id='wikcnKQ1k3pcuo5uSK4t8VN6kVf')
>>> wiki_fs.ls('/')
>>> # link-based read: pass space_id='dynamic', no space_id required for reading
>>> wiki_dynamic = FeishuFS(app_id='cli_xxx', app_secret='xxx', space_id='dynamic')
>>> content = wiki_dynamic.fetch_url('https://xxx.feishu.cn/wiki/MCjOwGxwSimztPkO5X6cv8uxnwb')
>>> # FS convenience: bare feishu URL is auto-detected and routed to wiki
>>> lazyllm.globals.config['feishu_wiki_space_id'] = 'wikcnKQ1k3pcuo5uSK4t8VN6kVf'  # for ls
>>> text = FS.read_bytes('https://xxx.feishu.cn/wiki/MCjOwGxwSimztPkO5X6cv8uxnwb').decode()
>>> text = FS.read_bytes('feishu:/~node/MCjOwGxwSimztPkO5X6cv8uxnwb').decode()
>>> text = FS.read_bytes('feishu@dynamic:/~node/MCjOwGxwSimztPkO5X6cv8uxnwb').decode()
''')

_add_feishu_chinese('FeishuFS.ls', '''\
列出飞书云盘指定路径下的文件和文件夹。

Args:
    path (str): 云盘目录路径，'/' 表示根目录。
    detail (bool): True 时返回详情 dict 列表；False 时仅返回名称列表。
''')
_add_feishu_english('FeishuFS.ls', '''\
List files and folders at the given path in Feishu Drive.

Args:
    path (str): Drive directory path; '/' for root.
    detail (bool): If True return list of dicts; if False return list of names.
''')

_add_feishu_chinese('FeishuFS.info', '''\
获取飞书云盘指定路径的文件或文件夹元信息。

Args:
    path (str): 文件或文件夹路径。
''')
_add_feishu_english('FeishuFS.info', '''\
Get metadata for a file or folder at the given path in Feishu Drive.

Args:
    path (str): File or folder path.
''')

_add_feishu_chinese('FeishuFS.mkdir', '''\
在飞书云盘中创建文件夹。

Args:
    path (str): 要创建的文件夹路径，最后一段为文件夹名，前面各段为父路径。
    create_parents (bool): 是否递归创建父目录（当前实现仅创建最后一级）。
''')
_add_feishu_english('FeishuFS.mkdir', '''\
Create a folder in Feishu Drive.

Args:
    path (str): Path of the folder to create; the last segment is the folder name, preceding segments are the parent path.
    create_parents (bool): Whether to create parent directories recursively (current implementation creates only the last level).
''')

_add_feishu_chinese('FeishuFS.move', '''\
将飞书云盘中的文件或文件夹移动到目标路径。

Args:
    path1 (str): 源文件或文件夹路径。
    path2 (str): 目标路径，前面各段为目标父路径。
    recursive (bool): 是否递归移动（文件夹移动时一并移动子项）。
''')
_add_feishu_english('FeishuFS.move', '''\
Move a file or folder to the target path in Feishu Drive.

Args:
    path1 (str): Source file or folder path.
    path2 (str): Destination path; preceding segments are the target parent path.
    recursive (bool): Whether to move recursively (folder move includes children).
''')

_add_feishu_chinese('FeishuFS.copy', '''\
复制飞书云盘中的文件到目标路径（不支持文件夹复制）。

Args:
    path1 (str): 源文件路径。
    path2 (str): 目标路径，最后一段为新文件名，前面各段为目标父路径。
    recursive (bool): 是否递归复制（文件夹不支持，传 True 会抛出 NotImplementedError）。
''')
_add_feishu_english('FeishuFS.copy', '''\
Copy a file to the target path in Feishu Drive (folder copy is not supported).

Args:
    path1 (str): Source file path.
    path2 (str): Destination path; the last segment is the new file name, preceding segments are the target parent path.
    recursive (bool): Whether to copy recursively (raises NotImplementedError for folders).
''')

# FeishuWikiFS
_add_fs_chinese('FeishuWikiFS', f'''\
飞书知识库文件系统：使用飞书开放平台 Wiki API，将指定知识库（space）映射为文件系统。目录对应 Wiki 目录/节点，文件对应文档或附件。支持 ls、info、open（读/写）、mkdir、rm_file、put_file、fetch_url；并支持文档块级文本编辑：get_document_id、get_doc_blocks、update_doc_block_text。

{_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH}

使用方式:
    构造参数:
        base_url (str, optional): 飞书开放平台根地址，默认使用官方地址即可。
        app_id (str, optional): 企业自建应用的 App ID。
        app_secret (str, optional): 企业自建应用的 App Secret。
        space_id (str): 知识空间 ID（如 'wikcnKQ1k3pcuo5uSK4t8VN6kVf'），或传入 'dynamic' 以懒解析。

    1. 与 FeishuFS 共享同一套鉴权方式：推荐 app_id + app_secret，由 FS 自动获取 tenant_access_token。
    2. space_id 解析优先级（每次需要时按序查找）：
       a. 构造时传入的真实 space_id；
       b. globals.config['feishu_wiki_space_id']（运行时设置）；
       c. 调用 get_node 接口时从响应回填（仅读文档时可省略 space_id）。
       ls/mkdir/copy/move 等树形操作必须能解析到有效 space_id，否则抛出 ValueError。
    3. 根路径 '/' 对应知识库根节点；路径为 Wiki 节点路径；open 会根据节点类型拉取文档纯文本或附件二进制；put_file 会新建 docx 并写入内容。当远程路径以 .md 结尾或 put_file(..., content_type='markdown') 时，会按 Markdown 解析并保持标题、列表、代码块、引用等格式；否则按纯文本追加段落。

链接直读路径（只读，无需 space_id）:
    支持以下路径前缀，直接通过飞书 token 或链接拉取内容，不需要标题路径或 space_id：
    - ~node/<node_token>       wiki 节点 token，如 feishu:/~node/MCjOwG...
    - ~link/<urlencoded_url>   飞书浏览器链接（URL 编码），如 feishu:/~link/https%3A%2F%2F...
    - ~docx/<document_id>      docx 文档 id
    - ~doc/<doc_token>         旧版 doc token
    也可以直接传入裸 URL（feishu:/ 协议会自动识别 feishu.cn / larksuite.com 链接）。
    写模式（open write / put_file）不支持 ~ 路径，会抛出 NotImplementedError。

保持原格式（表格、标题等）:
    - 通过 open/raw_content 下载到的是纯文本，put_file 会新建文档并只追加段落，因此「下载-修改-上传」会丢失表格等格式。
    - get_doc_blocks(path) 返回飞书 Block API 的原始 block 字段，包括 elements、style、children、表格属性和未知类型字段；每个 block 额外带有便于检索的派生 plain_text。可对需要修改的文本块调用 update_doc_block_text(path, block_id, new_text)，其它 block 不会被改动。
''')
_add_fs_english('FeishuWikiFS', f'''\
Feishu Wiki FS: Feishu Open Platform Wiki API; maps a wiki space into a filesystem. Directories are wiki folders/nodes; files are documents or attachments. Supports ls, info, open (r/w), mkdir, rm_file, put_file, fetch_url, and block-level text edit: get_document_id, get_doc_blocks, update_doc_block_text.

{_FEISHU_DOCUMENT_LINK_WORKFLOW_EN}

Usage:
    1. Shares the same auth model as FeishuFS; space_id is a wiki space ID or 'dynamic' for lazy resolution.
    2. space_id resolution order (checked on each operation):
       a. Real space_id passed to the constructor;
       b. globals.config['feishu_wiki_space_id'] (set at runtime);
       c. Backfilled from the get_node response (no space_id required for link-based reads).
       Tree operations (ls/mkdir/copy/move) require a resolvable space_id; otherwise ValueError is raised.
    3. '/' is the wiki root; paths are wiki node paths; open returns document plain text or file binary; put_file creates a new docx. When the remote path ends with .md or put_file(..., content_type='markdown'), content is parsed as Markdown and rendered as headings, lists, code blocks, quotes, etc.; otherwise appended as plain text paragraphs.

Link-based read paths (read-only, no space_id required):
    The following path prefixes bypass title-based resolution and fetch content directly by token:
    - ~node/<node_token>       wiki node token, e.g. feishu:/~node/MCjOwG...
    - ~link/<urlencoded_url>   URL-encoded Feishu browser link, e.g. feishu:/~link/https%3A%2F%2F...
    - ~docx/<document_id>      docx document id
    - ~doc/<doc_token>         legacy doc token
    Bare Feishu URLs (feishu.cn / larksuite.com) are also accepted and routed automatically.
    Write mode (open write / put_file) on ~ paths raises NotImplementedError.

Preserving format (tables, headings, etc.):
    - open/raw_content returns plain text only; put_file creates a new doc and appends paragraphs, so download-modify-upload loses tables and other structure.
    - get_doc_blocks(path) returns the native Feishu Block API fields, including elements, styles, children, table properties, and unknown block-type fields; each block also includes a derived plain_text value for search. Use update_doc_block_text(path, block_id, new_text) for targeted text changes; other blocks are left unchanged.
''')
_add_fs_chinese('FeishuWikiFS.fetch_url', f'''\
通过飞书浏览器链接直接拉取文档内容（只读），无需 space_id 也无需标题路径。已废弃，推荐使用 read_bytes(url) 或 open(url)。

{_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH}

支持的链接格式：
    - https://<host>/wiki/<node_token>   知识库节点链接
    - https://<host>/docx/<document_id>  新版文档直链
    - https://<host>/docs/<doc_token>    旧版文档直链
host 支持 *.feishu.cn 和 *.larksuite.com，query 参数会被忽略。

Args:
    url (str): 飞书浏览器链接。

Returns:
    bytes: 文档纯文本内容（UTF-8 编码）；附件类型返回二进制。
''')
_add_fs_english('FeishuWikiFS.fetch_url', f'''\
Fetch document content directly from a Feishu browser URL (read-only), without requiring space_id or a title path. Deprecated; prefer read_bytes(url) or open(url).

{_FEISHU_DOCUMENT_LINK_WORKFLOW_EN}

Supported URL formats:
    - https://<host>/wiki/<node_token>   wiki node link
    - https://<host>/docx/<document_id>  new-style docx direct link
    - https://<host>/docs/<doc_token>    legacy doc direct link
host supports *.feishu.cn and *.larksuite.com; query parameters are ignored.

Args:
    url (str): Feishu browser URL.

Returns:
    bytes: Document plain text content (UTF-8 encoded); binary for file attachments.
''')
_add_fs_chinese('FeishuWikiFS.read_bytes', f'''\
读取路径对应文档的完整内容（字节）。支持裸飞书 URL、~link/~node/~docx/~doc 路径及标题路径。

{_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    path (str): 文档路径或飞书 URL。
    include_references (bool): 为 True 时在正文末尾追加引用附录（lazyllm-feishu-references 格式），默认 False。
''')
_add_fs_english('FeishuWikiFS.read_bytes', f'''\
Read the full content of the document at path as bytes. Supports bare Feishu URLs, ~link/~node/~docx/~doc paths, and title paths.

{_FEISHU_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    path (str): Document path or Feishu URL.
    include_references (bool): When True, appends a reference appendix (lazyllm-feishu-references format) after the body; default False.
''')

_add_fs_chinese('FeishuWikiFS.read', f'''\
读取路径对应文档的完整内容并以 UTF-8 字符串返回。支持裸飞书 URL、~link/~node/~docx/~doc 路径及标题路径。

{_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    path (str): 文档路径或飞书 URL。

Returns:
    str: 文档文本内容（UTF-8 解码）。
''')
_add_fs_english('FeishuWikiFS.read', f'''\
Read the full content of the document at path and return as a UTF-8 string. Supports bare Feishu URLs, ~link/~node/~docx/~doc paths, and title paths.

{_FEISHU_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    path (str): Document path or Feishu URL.

Returns:
    str: Document text content (UTF-8 decoded).
''')
_add_fs_chinese('FeishuWikiFS.ls', '''\
列出路径下的子节点。支持裸飞书 wiki URL、~link/<encoded_url>、~node/<token> 路径及标题路径。

- 传入 wiki URL 或 ~node 路径时，列举该节点的直接子节点（不拉取子文档正文）。
- 节点无子页（has_child=false）时返回空列表，不报错。
- 传入 docx/docs 直链（非 wiki 节点）时抛出 ValueError。

Args:
    path (str): 目录路径、wiki URL 或 ~node/~link 路径。
    detail (bool): True 时返回 dict 列表（含 creator/owner/node_token 等）；False 时返回名称列表。
''')
_add_fs_english('FeishuWikiFS.ls', '''\
List child nodes at path. Supports bare Feishu wiki URLs, ~link/<encoded_url>, ~node/<token> paths, and title paths.

- For wiki URLs or ~node paths, lists direct child nodes (does not fetch document content).
- Returns empty list when the node has no children (has_child=false); does not raise.
- Raises ValueError for docx/docs direct links (non-wiki nodes).

Args:
    path (str): Directory path, wiki URL, or ~node/~link path.
    detail (bool): If True return list of dicts (with creator/owner/node_token etc.); if False return list of names.
''')
_add_fs_chinese('FeishuWikiFS.info', '''\
获取路径对应节点的元信息。支持裸飞书 wiki URL、~link/~node 路径及标题路径。

Args:
    path (str): 节点路径、wiki URL 或 ~node/~link 路径。
''')
_add_fs_english('FeishuWikiFS.info', '''\
Get metadata for the node at path. Supports bare Feishu wiki URLs, ~link/~node paths, and title paths.

Args:
    path (str): Node path, wiki URL, or ~node/~link path.
''')
_add_feishu_chinese('FeishuWikiFS.mkdir', '''\
在知识库中创建新的 docx 节点（目录/文档）。

Args:
    path (str): 要创建的节点路径，最后一段为节点标题，前面各段为父路径。
    create_parents (bool): 是否递归创建父节点（当前实现仅创建最后一级）。
''')
_add_feishu_english('FeishuWikiFS.mkdir', '''\
Create a new docx node (directory/document) in the wiki space.

Args:
    path (str): Path of the node to create; the last segment is the node title, preceding segments are the parent path.
    create_parents (bool): Whether to create parent nodes recursively (current implementation creates only the last level).
''')

_add_feishu_chinese('FeishuWikiFS.move', '''\
将知识库节点移动到目标路径。

Args:
    path1 (str): 源节点路径。
    path2 (str): 目标路径，最后一段为新标题，前面各段为目标父路径。
    recursive (bool): 是否递归移动（当前实现不区分，节点含子节点时一并移动）。
''')
_add_feishu_english('FeishuWikiFS.move', '''\
Move a wiki node to the target path.

Args:
    path1 (str): Source node path.
    path2 (str): Destination path; the last segment is the new title, preceding segments are the target parent path.
    recursive (bool): Whether to move recursively (current implementation moves the node including its children).
''')

_add_feishu_chinese('FeishuWikiFS.copy', '''\
复制知识库节点到目标路径。

Args:
    path1 (str): 源节点路径。
    path2 (str): 目标路径，最后一段为新标题，前面各段为目标父路径。
    recursive (bool): 是否递归复制（当前实现不区分，节点含子节点时一并复制）。
''')
_add_feishu_english('FeishuWikiFS.copy', '''\
Copy a wiki node to the target path.

Args:
    path1 (str): Source node path.
    path2 (str): Destination path; the last segment is the new title, preceding segments are the target parent path.
    recursive (bool): Whether to copy recursively (current implementation copies the node including its children).
''')
_add_feishu_chinese('FeishuWikiFS.resolve_wiki_ref', f'''\
解析飞书 wiki URL、~node/~link 路径、裸飞书 URL 或标题路径，返回标准化节点元信息。用于 chat 中用户粘贴飞书链接后确定节点类型、标题和 ID，再读取正文。自动区分 wiki_node、docx、doc 类型。

{_FEISHU_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    url_or_path (str): 飞书 wiki URL、~node/~link/~docx/~doc 路径或知识库标题路径。

Returns:
    Dict[str, Any]: 包含 node_token、space_id、title、obj_type、obj_token、has_child、creator、owner、node_creator 等字段。
''')
_add_feishu_english('FeishuWikiFS.resolve_wiki_ref', f'''\
Resolve a Feishu wiki URL, ~node/~link path, bare Feishu URL, or title path into normalized node metadata. Use after users paste Feishu links to identify node type, title, and id before reading content. Automatically distinguishes wiki_node, docx, and doc types.

{_FEISHU_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    url_or_path (str): Feishu wiki URL, ~node/~link/~docx/~doc path, or wiki title path.

Returns:
    Dict[str, Any]: Metadata including node_token, space_id, title, obj_type, obj_token, has_child, creator, owner, and node_creator.
''')

_add_fs_chinese('FeishuWikiFS.get_document_id', '''\
解析飞书或 Lark 浏览器文档链接或 Wiki 标题路径并返回对应的 document_id。docx/docs 直链直接返回链接中的 token；Wiki 链接和标题路径会查询节点并返回其 obj_token。非飞书链接或非文档节点会抛出 ValueError。

Args:
    path (str): 飞书 docx、docs、Wiki 文档链接或 Wiki 标题路径；也接受 FS Router 生成的 ~link 路径。不支持 AppLink 和短链。

Returns:
    str: 文档的 document_id，用于飞书 docx API。
''')
_add_fs_english('FeishuWikiFS.get_document_id', '''\
Resolve a Feishu or Lark browser document URL or Wiki title path and return its document_id. Direct docx/docs URLs return their token; Wiki URLs and title paths resolve the node and return its obj_token. Non-Feishu URLs and non-document nodes raise ValueError.

Args:
    path (str): Feishu docx, docs, or Wiki document URL, or a Wiki title path. FS Router ~link paths are also accepted. AppLinks and short URLs are not supported.

Returns:
    str: The document_id for Feishu docx API.
''')
_add_fs_chinese('FeishuWikiFS.get_doc_blocks', '''\
获取文档的块列表（block 树扁平列表）。用于在编辑时定位要修改的块并保留表格等非文本块。

Args:
    path (str): Wiki 内文档路径。
    with_descendants (bool): 是否包含所有子孙块，默认 True。

Returns:
    list: 每项为飞书 Block API 的原始 dict，完整保留 elements、style、children、表格属性及未知字段，并附加派生 plain_text。
''')
_add_fs_english('FeishuWikiFS.get_doc_blocks', '''\
Get the document block list (flattened block tree). Use to locate blocks to edit while leaving tables and other non-text blocks unchanged.

Args:
    path (str): Wiki path to the document.
    with_descendants (bool): Whether to include all descendant blocks; default True.

Returns:
    List[Dict[str, Any]]: Native Feishu Block API dictionaries with elements, styles, children, table properties, and unknown fields preserved, plus a derived plain_text value.
''')
_add_fs_chinese('FeishuWikiFS.update_doc_block_text', '''\
更新文档中指定块的文本内容。仅适用于支持文本的块（如 Text、Heading、Bullet 等）；表格等块不支持，调用会由飞书 API 报错。

Args:
    path (str): Wiki 内文档路径。
    block_id (str): 要更新的块 ID（来自 get_doc_blocks 返回的 block_id）。
    new_text (str): 新的纯文本内容。
''')
_add_fs_english('FeishuWikiFS.update_doc_block_text', '''\
Update the text content of a specific block in the document. Only applies to text-capable blocks (e.g. Text, Heading, Bullet); table blocks are not supported and the Feishu API will return an error.

Args:
    path (str): Wiki path to the document.
    block_id (str): Block ID to update (from get_doc_blocks).
    new_text (str): New plain text content.
''')

_add_fs_chinese('FeishuWikiFS.search', '''\
在飞书知识库中按关键词搜索节点，使用飞书官方 wiki/v2/nodes/search 接口。
搜索范围包括节点标题和正文内容，返回当前用户可见的匹配 wiki 节点。

这是在线搜索 —— 直接查询飞书线上知识库，不是本地已入库的文档。

Args:
    query (str or List[str]): 一个搜索词、以空格分隔的多个词，或搜索词列表。
    space_id (str, optional): 要搜索的知识空间 ID；为空时搜索当前用户可见的全部 Wiki。
    node_id (str, optional): 将范围限定到指定节点及其子节点；使用时必须同时提供 space_id。
    page_size (int): 最大返回条数，默认 20，最大 50。

Returns:
    List[Dict[str, Any]]: 匹配节点列表。每项包含 title、node_token、obj_type、url、space_id。
''')
_add_fs_english('FeishuWikiFS.search', '''\
Search wiki nodes by keyword using Feishu's official wiki/v2/nodes/search API.
Matches node titles and content visible to the current user.

This searches the LIVE online Feishu wiki — not locally indexed documents.

Args:
    query (str or List[str]): One term, multiple space-separated terms, or a list of terms.
    space_id (str, optional): Wiki space ID to search; when empty, searches all Wiki spaces visible to the user.
    node_id (str, optional): Limit the search to a node and its descendants; requires space_id.
    page_size (int): Maximum results, default 20, maximum 50.

Returns:
    List[Dict[str, Any]]: Matching nodes, each with title, node_token, obj_type, url, and space_id.
''')

_add_fs_chinese('FeishuWikiFS.find', '''\
按文件名/标题正则匹配查找知识库节点。只匹配节点标题（文件名），不搜索正文内容。

通过递归遍历 wiki 树并逐标题做正则筛选实现；默认大小写不敏感。

常用正则示例：
- "report" 匹配标题含 report 的节点
- "^2024" 匹配以 2024 开头的标题
- "(设计|方案)" 匹配含"设计"或"方案"的标题

Args:
    pattern (str): 正则表达式模式，大小写不敏感。
    space_id (str, optional): 要搜索的知识空间 ID；为空时遍历飞书“获取知识空间列表”接口返回的空间。
        该官方接口不返回“我的文档库”，查找个人库时应显式传入 space_id。
    max_results (int): 最大返回条数，默认 50，最大 200。

Returns:
    List[Dict[str, Any]]: 匹配节点列表。每项包含 title、node_token、obj_type、url、
    space_id、has_child。
''')
_add_fs_english('FeishuWikiFS.find', '''\
Find wiki nodes by filename/title matching a regex pattern. Matches only node titles (names),
not document content.

Implemented by recursively listing the wiki tree and filtering by title regex; case-insensitive
by default.

Common regex examples:
- "report" matches titles containing "report"
- "^2024" matches titles starting with "2024"
- "(design|spec)" matches titles with either "design" or "spec"

Args:
    pattern (str): Regular expression pattern, case-insensitive.
    space_id (str, optional): Wiki space ID to search. When empty, walks spaces returned by Feishu's list-spaces API.
        That official API excludes My Library, so pass space_id explicitly for a personal Wiki.
    max_results (int): Maximum results, default 50, capped at 200.

Returns:
    List[Dict[str, Any]]: Matching nodes, each with title, node_token, obj_type, url,
    space_id, and has_child.
''')

# ConfluenceFS
_add_fs_chinese('ConfluenceFS', '''\
Confluence 文件系统：基于 Atlassian Confluence REST API，以 Space/Page 为目录与文件。目录监听用 CloudFsWatchdog；支持 webhook。

认证与配置: 构造参数 token、email、cloud_id；未传时在 ConfluenceFS 内从 config（confluence_token、confluence_email、confluence_cloud_id，环境变量 CONFLUENCE_TOKEN、CONFLUENCE_EMAIL、CONFLUENCE_CLOUD_ID）读取。

如何获取 token:
    1. 登录 Confluence（或 Atlassian 账号），点击头像 → Account settings（或 设置）。
    2. 在 Security 区域找到「Create and manage API tokens」或「API tokens」，点击进入。
    3. 点击「Create API token」，输入标签名后创建，复制生成的 token（仅显示一次），作为本 FS 的 token。
    4. 云版 Confluence（cloud=True）: 需同时提供登录邮箱（email）和上述 API token，用于 Basic 认证；cloud_id 可从 Confluence 云实例的 URL 或 Atlassian 的「API 开发」文档中获取（格式如 站点 id）。
''')
_add_fs_english('ConfluenceFS', '''\
Confluence FS: Atlassian Confluence REST API; Space/Page as dir/file. Use CloudFsWatchdog for watching; webhook supported.

Auth and config: Constructor token, email, cloud_id; when not passed, resolved inside ConfluenceFS from config (confluence_token, confluence_email, confluence_cloud_id; env CONFLUENCE_TOKEN, CONFLUENCE_EMAIL, CONFLUENCE_CLOUD_ID).

How to obtain token:
    1. Log in to Confluence (or Atlassian account), go to Profile → Account settings.
    2. Under Security, open "Create and manage API tokens" (or "API tokens").
    3. Click "Create API token", name it, then copy the generated token (shown once) and use it as token for this FS.
    4. Confluence Cloud (cloud=True): Also provide your login email and the API token for Basic auth; get cloud_id from your Cloud instance URL or Atlassian API docs (site id).
''')
_add_fs_example('ConfluenceFS', '''\
>>> from lazyllm.tools.fs import ConfluenceFS
>>> fs = ConfluenceFS(token='xxx', email='user@example.com', cloud=True, cloud_id='xxx')
>>> fs.ls('/SPACE')
''')

# NotionFS
_add_notion_chinese('NotionFile', '''\
Notion 页面对应的只读缓冲文件对象。打开页面时会先通过 NotionFS 读取页面 Markdown 内容，
再提供 fsspec 风格的区间读取能力，供上层统一按文件对象消费。
''')
_add_notion_english('NotionFile', '''\
Read-only buffered file object for a Notion page. It preloads page Markdown through NotionFS and
then exposes fsspec-style range reads so upper layers can consume it as a file object.
''')

_add_notion_chinese('NotionFile.__init__', '''\
创建用于 Notion 页面读取的缓冲文件对象。构造时会通过所属 NotionFS 预取页面内容，并按 fsspec 的 range read 接口提供给上层。

Args:
    fs (NotionFS): 所属 Notion 文件系统实例。
    path (str): 页面、数据库或 block 路径。
    include_references (bool): 是否在读取内容末尾追加 Notion 引用页脚。
    **kwargs: 透传给 CloudFSBufferedFile 的 fsspec 文件参数。
''')
_add_notion_english('NotionFile.__init__', '''\
Create a buffered file object for reading Notion page content. It prefetches content through the owning NotionFS and exposes it through fsspec-style range reads.

Args:
    fs (NotionFS): Owning Notion filesystem.
    path (str): Page, database, or block path.
    include_references (bool): Whether to append a Notion references footer to the content.
    **kwargs: fsspec file options forwarded to CloudFSBufferedFile.
''')

_add_fs_chinese('NotionFS', f'''\
Notion 文件系统：基于 Notion API，以 Page/Block 为层级，支持 ls、读写、mkdir、rm（归档）。
写入页面内容时，Notion API 单 block 的 rich_text 限制为 2000 字符，超长内容会被截断。

{_NOTION_DOCUMENT_LINK_WORKFLOW_ZH}

认证与配置: 构造参数 token（Notion Integration Token / Internal Integration Secret）；可选 base_url。
环境变量: NOTION_TOKEN、NOTION_API_KEY，任一非空即可作为 token。

如何获取 token:
    1. 登录 https://www.notion.so，进入要管理的 Workspace。
    2. 侧栏底部点击「Settings & members」→「Connections」或「Integrations」。
    3. 点击「Develop or manage integrations」或「New integration」，创建新集成（Integration）。
    4. 在集成详情页复制「Internal Integration Secret」或「API key」（形如 secret_xxx），即为本 FS 的 token。注意：需在需要访问的页面/数据库中，通过「Connections」或「Add connections」将该集成连接上，否则 API 无法访问该内容。
''')
_add_fs_english('NotionFS', f'''\
Notion FS: Notion API; Page/Block hierarchy; supports ls, read/write, mkdir, rm (archive).
When writing page content, Notion API limits rich_text to 2000 chars per block; longer content is truncated.

{_NOTION_DOCUMENT_LINK_WORKFLOW_EN}

Auth and config: Constructor token (Notion Integration Token / Internal Integration Secret); optional base_url.
Env vars: NOTION_TOKEN, NOTION_API_KEY; any non-empty used as token.

How to obtain token:
    1. Log in at https://www.notion.so and open the target Workspace.
    2. Go to Settings & members → Connections (or Integrations).
    3. Click "Develop or manage integrations" or "New integration" and create an integration.
    4. Copy the "Internal Integration Secret" (or "API key", e.g. secret_xxx) from the integration page; that is the token for this FS. You must connect this integration to each page/database you want to access (via "Connections" / "Add connections" on the page), or the API cannot see that content.
''')
_add_fs_example('NotionFS', '''\
>>> from lazyllm.tools.fs import NotionFS
>>> fs = NotionFS(token='xxx')
>>> fs.ls('/')
''')
_add_fs_chinese('NotionFS.ls', '''\
列出当前 token 可访问的 Notion 页面、数据库或指定对象的子项。传入根路径 ``/`` 时，
通过 Notion 官方搜索接口列出所有可访问对象；聊天工具调用中省略 path 或传入空路径时，
也会按根路径处理。

Args:
    path (str): Notion 路径、对象 ID、浏览器 URL 或 notion:/ URI；默认为根路径 ``/``。
    detail (bool): 是否返回完整元信息；为 False 时仅返回名称。

Returns:
    List: 可访问对象或子项列表。
''')
_add_fs_english('NotionFS.ls', '''\
List Notion pages, databases, or children of a specific object visible to the current token.
For the root path ``/``, this uses Notion's official search endpoint to list all accessible
objects. Chat tool calls that omit path or provide an empty path are also treated as root.

Args:
    path (str): Notion path, object id, browser URL, or notion:/ URI; defaults to root ``/``.
    detail (bool): Whether to return full metadata; when False, return names only.

Returns:
    List: Accessible objects or child entries.
''')
_add_fs_chinese('NotionFS.read_bytes', f'''\
读取 Notion 页面、数据库或 block 路径的完整内容，并返回 UTF-8 字节。支持 Notion 浏览器 URL、notion:/ URI、页面/数据库/block ID 或路径。

{_NOTION_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    path (str): Notion 页面、数据库或 block 路径，也可以是 Notion 浏览器链接或 notion:/ URI；会容忍常见展示态包裹，如反引号、加粗标记、尖括号或 Markdown 链接。
    include_references (bool): 为 True 时在正文末尾追加 Notion 引用页脚，默认 False。

Returns:
    bytes: 页面内容的 Markdown/文本字节。
''')
_add_fs_english('NotionFS.read_bytes', f'''\
Read the full content for a Notion page, database, or block path and return UTF-8 bytes. Supports Notion browser URLs, notion:/ URIs, page/database/block ids, or paths.

{_NOTION_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    path (str): Notion page, database, or block path; may also be a Notion browser link or notion:/ URI. Common display wrappers such as backticks, bold markers, angle brackets, or Markdown links are tolerated.
    include_references (bool): When True, appends a Notion references footer after the body; default False.

Returns:
    bytes: Markdown/text bytes for the page content.
''')
_add_fs_chinese('NotionFS.search', '''\
按标题搜索当前 token 可访问的 Notion 页面或数据库。该能力来自 Notion 官方 /v1/search 接口，主要用于定位资源；不是页面正文全文检索。

Args:
    query (str): 标题关键词。
    object_type (str): 可选对象过滤，支持 page、database。
    limit (int): 最大返回条数，默认 20，最大 100。
    sort_direction (str): 按 last_edited_time 排序方向，ascending 或 descending。
    scope (str): 可选 Notion database 或 data_source 范围，支持 notion:/~database/<id>、notion:/~data_source/<id>、database:<id>、data_source:<id>。
    title_pattern (str): 可选标题/文件名正则过滤。

Returns:
    List[Dict[str, Any]]: 搜索结果条目，包含 title、id、notion_path、url 等字段。
''')
_add_fs_english('NotionFS.search', '''\
Search Notion pages or databases visible to the current token by title. This uses Notion's official /v1/search endpoint to locate resources; it is not full-text page-body search.

Args:
    query (str): Title keyword.
    object_type (str): Optional object filter: page or database.
    limit (int): Maximum number of results, default 20, capped at 100.
    sort_direction (str): Sort direction by last_edited_time: ascending or descending.
    scope (str): Optional Notion database or data_source scope, such as notion:/~database/<id>, notion:/~data_source/<id>, database:<id>, or data_source:<id>.
    title_pattern (str): Optional title/filename regex filter.

Returns:
    List[Dict[str, Any]]: Search result entries with title, id, notion_path, url, and related metadata.
''')

_add_fs_chinese('NotionFS.find', '''\
按页面/数据库标题正则匹配查找 Notion 对象。只匹配标题（名称），不搜索页面正文内容。

使用 Notion 官方 /v1/search 接口做宽泛查询后，在客户端按标题正则筛选。默认大小写不敏感。

常用正则示例：
- "PRD" 匹配标题含 PRD 的页面
- "^Q[1-4]" 匹配以 Q1-Q4 开头的标题
- "(OKR|KPI)" 匹配含 OKR 或 KPI 的标题

Args:
    pattern (str): 正则表达式模式，大小写不敏感。
    object_type (str, optional): 对象类型过滤：''（全部）、page、database。
    limit (int): 最大返回条数，默认 50，最大 100。
    scope (str): 可选 Notion database 或 data_source 范围，支持 notion:/~database/<id>、notion:/~data_source/<id>、database:<id>、data_source:<id>。

Returns:
    List[Dict[str, Any]]: 匹配的条目列表。每项包含 title/name、id、type（file/directory）、
    notion_path 等字段。
''')
_add_fs_english('NotionFS.find', '''\
Find Notion pages or databases by title matching a regex pattern. Matches only titles (names),
not page body content.

Uses Notion's official /v1/search API for broad lookup, then filters by title regex client-side.
Case-insensitive by default.

Common regex examples:
- "PRD" matches pages with "PRD" in the title
- "^Q[1-4]" matches titles starting with Q1 through Q4
- "(OKR|KPI)" matches titles containing either OKR or KPI

Args:
    pattern (str): Regular expression pattern, case-insensitive.
    object_type (str, optional): Object type filter: '' (all), page, or database.
    limit (int): Maximum results, default 50, capped at 100.
    scope (str): Optional Notion database or data_source scope, such as notion:/~database/<id>, notion:/~data_source/<id>, database:<id>, or data_source:<id>.

Returns:
    List[Dict[str, Any]]: Matching entries, each with title/name, id, type (file/directory),
    and notion_path.
''')

_add_fs_chinese('NotionFS.replace_page_markdown', '''\
使用 Notion Markdown endpoint 替换页面正文。该接口适合把算法生成的整页 Markdown 写回 Notion；是否允许删除原内容由 allow_deleting_content 控制。

Args:
    page_id (str): Notion 页面 ID，可为带横线或不带横线格式。
    markdown (str): 新的 Markdown 正文。
    allow_deleting_content (bool): 是否允许删除页面已有内容，默认 False。

Returns:
    Dict[str, Any]: Notion API 返回体。
''')
_add_fs_english('NotionFS.replace_page_markdown', '''\
Replace a page body through Notion's Markdown endpoint. This is useful when algorithm output should be written back as full-page Markdown; allow_deleting_content controls whether existing content may be removed.

Args:
    page_id (str): Notion page id, hyphenated or compact.
    markdown (str): New Markdown body.
    allow_deleting_content (bool): Whether existing page content may be deleted. Defaults to False.

Returns:
    Dict[str, Any]: Notion API response payload.
''')

_add_fs_chinese('NotionFS.insert_page_markdown', '''\
向页面插入 Markdown 内容。默认插入到页面末尾，适合追加总结、分析结论或同步生成的段落。

Args:
    page_id (str): Notion 页面 ID，可为带横线或不带横线格式。
    markdown (str): 要插入的 Markdown 内容。
    position (str): 插入位置，默认 end。

Returns:
    Dict[str, Any]: Notion API 返回体。
''')
_add_fs_english('NotionFS.insert_page_markdown', '''\
Insert Markdown content into a page. By default the content is appended to the end, which fits summaries, analysis notes, or generated paragraphs.

Args:
    page_id (str): Notion page id, hyphenated or compact.
    markdown (str): Markdown content to insert.
    position (str): Insert position. Defaults to end.

Returns:
    Dict[str, Any]: Notion API response payload.
''')

_add_fs_chinese('NotionFS.update_page_title', '''\
更新 Notion 页面的标题属性。方法会自动识别页面的 title 类型属性，再通过 pages/{page_id} PATCH 写入。

Args:
    page_id (str): Notion 页面 ID，可为带横线或不带横线格式。
    title (str): 新标题文本。
''')
_add_fs_english('NotionFS.update_page_title', '''\
Update the title property of a Notion page. The method detects the page's title property and patches pages/{page_id}.

Args:
    page_id (str): Notion page id, hyphenated or compact.
    title (str): New title text.
''')

_add_fs_chinese('NotionFS.resolve_notion_ref', f'''\
解析 Notion URL、notion:/ URI 或页面/数据库/block ID，并返回统一元信息。用于 chat 中用户粘贴链接后先确定对象类型、标题和可读路径。

{_NOTION_DOCUMENT_LINK_WORKFLOW_ZH}

Args:
    url_or_path (str): Notion 浏览器链接、notion:/ URI、页面/数据库/block ID 或路径；会容忍常见展示态包裹，如反引号、加粗标记、尖括号或 Markdown 链接。

Returns:
    Dict[str, Any]: 包含 object_id、object_type、title、notion_path、has_child 等字段的元信息。
''')
_add_fs_english('NotionFS.resolve_notion_ref', f'''\
Resolve a Notion URL, notion:/ URI, or page/database/block id into normalized metadata. This lets chat flows identify object type, title, and readable path after users paste links.

{_NOTION_DOCUMENT_LINK_WORKFLOW_EN}

Args:
    url_or_path (str): Notion browser link, notion:/ URI, page/database/block id, or path. Common display wrappers such as backticks, bold markers, angle brackets, or Markdown links are tolerated.

Returns:
    Dict[str, Any]: Metadata including object_id, object_type, title, notion_path, and has_child.
''')

# GoogleDriveFS
_add_fs_chinese('GoogleDriveFS', '''\
Google Drive 文件系统：使用 Google 官方 Drive v3 API 直接访问在线原始云盘，支持目录浏览、文件读写、关键词搜索和文件名正则查找。不会先把文件导入 LazyLLM 知识库。

构造参数:
    credentials (str | dict, optional): Service Account JSON 文件路径或已解析的 dict；提供后会自动换取 access_token。
    base_url (str, optional): Drive API 根地址，默认官方地址。
    dynamic_auth (bool): 是否由 Agent/ToolManager 在每次请求时注入用户 OAuth access token；默认 False。

认证模式:
    - 独立 Python 调用：传入 Service Account JSON 路径或 dict，FS 自动换取并刷新 access_token。
    - Agent/LazyMind 在线工具：使用 GoogleDriveFS(dynamic_auth=True)，由 ToolManager 按请求注入已授权用户的 OAuth access token。构造函数不接收 token 参数，请勿把 access_token 当作 credentials 传入。
    - drive.readonly 权限可用于 search、find 和读取；写入、移动、删除等操作需要更高的 Google Drive OAuth 权限，否则官方 API 会拒绝请求。

配置与环境变量: config 项 googledrive_credentials（环境变量 GOOGLE_APPLICATION_CREDENTIALS）指向 Service Account JSON 路径；在 GoogleDriveFS 内解析。

如何获取 OAuth / Service Account 凭证:
    方式 A — OAuth2（用户身份，由上层应用管理 token 生命周期）:
    1. 打开 Google Cloud Console https://console.cloud.google.com，创建或选择项目。
    2. 启用「Google Drive API」：APIs & Services → Enable APIs and Services → 搜索并启用 Drive API。
    3. 创建 OAuth 2.0 凭据：APIs & Services → Credentials → Create Credentials → OAuth client ID，应用类型选 Desktop 或 Web，获取 client_id 与 client_secret。
    4. 上层应用使用 Google 官方 OAuth 流程取得 access_token/refresh_token，并通过 dynamic_auth 工具调用注入短期 access_token。
    方式 B — Service Account（服务身份）:
    1. 同上在 Cloud Console 启用 Drive API。
    2. 创建服务账号：APIs & Services → Credentials → Create Credentials → Service account，创建后进入该服务账号 → Keys → Add key → JSON，下载 JSON 密钥文件。
    3. 将需要访问的 Google Drive 文件夹或「我的云端硬盘」共享给该服务账号的邮箱（如 xxx@yyy.iam.gserviceaccount.com），权限至少为「查看者」或「编辑者」。
    4. 构造 GoogleDriveFS 时传入 credentials 为该 JSON 文件路径或已解析的 dict，无需手动传 token（内部会用 JSON 中的私钥换取 access_token）。
''')
_add_fs_english('GoogleDriveFS', '''\
Google Drive filesystem backed directly by Google's official Drive v3 API. It browses, reads, writes, searches, and finds files in the live source Drive without importing them into a LazyLLM knowledge base first.

Args:
    credentials (str | dict, optional): Service Account JSON path or parsed dictionary. The filesystem obtains and refreshes its access token.
    base_url (str, optional): Drive API base URL; defaults to the official endpoint.
    dynamic_auth (bool): Let Agent/ToolManager inject a user OAuth access token per request. Defaults to False.

Authentication modes:
    - Standalone Python: pass a Service Account JSON path or dictionary.
    - Agent/LazyMind online tools: construct GoogleDriveFS(dynamic_auth=True); ToolManager injects the authorized user's OAuth access token for each request. The constructor has no token argument, so do not pass an access token as credentials.
    - drive.readonly is sufficient for search, find, and read. Write, move, and delete operations require broader Google Drive OAuth scopes and are rejected by the official API otherwise.

Config and env: config key googledrive_credentials (env GOOGLE_APPLICATION_CREDENTIALS) points to Service Account JSON path; resolved inside GoogleDriveFS.

How to obtain OAuth / Service Account credentials:
    Option A — OAuth2 (user identity; token lifecycle managed by the host application):
    1. In Google Cloud Console (https://console.cloud.google.com), create or select a project and enable the Google Drive API.
    2. Create OAuth 2.0 credentials (Desktop or Web client), get client_id and client_secret.
    3. Run Google's official OAuth flow in the host application, retain access_token/refresh_token there, and inject the short-lived access token through dynamic_auth tool execution.
    Option B — Service Account:
    1. Enable Drive API and create a Service Account in the same project; download its JSON key from Keys → Add key → JSON.
    2. Share the target Drive folder (or My Drive) with the service account email (e.g. xxx@yyy.iam.gserviceaccount.com) with at least Viewer/Editor access.
    3. Pass credentials as the JSON file path or parsed dict to GoogleDriveFS; no need to pass token (the class will obtain access_token from the key).
''')
_add_fs_example('GoogleDriveFS', '''\
>>> from lazyllm.tools.fs import GoogleDriveFS
>>> # Standalone use with a service account shared into the target Drive.
>>> fs = GoogleDriveFS(credentials='/path/to/service-account.json')
>>> fs.ls('/root')
>>> # Agent/ToolManager integrations construct with dynamic_auth=True and inject OAuth per request.
>>> online_fs = GoogleDriveFS(dynamic_auth=True)
''')

_add_fs_chinese('GoogleDriveFS.ls', '''\
列出 Google Drive 文件夹中的直接子项。使用官方 files.list API；路径为空时列出“我的云端硬盘”根目录，路径末段也可以直接使用文件夹 ID。共享盘路径支持 /drive/<drive_id>/<folder_id>。

Args:
    path (str): 文件夹路径或文件夹 ID。
    detail (bool): 为 True 时返回元数据字典；为 False 时仅返回名称。默认 True。
    **kwargs: 预留的文件系统参数。

Returns:
    List: 文件夹直接子项的元数据或名称列表。
''')
_add_fs_english('GoogleDriveFS.ls', '''\
List direct children of a Google Drive folder with the official files.list API. An empty path lists the My Drive root, and the final path segment may be a folder id. Shared-drive paths support /drive/<drive_id>/<folder_id>.

Args:
    path (str): Folder path or folder id.
    detail (bool): Return metadata dictionaries when True, or names only when False. Defaults to True.
    **kwargs: Reserved filesystem options.

Returns:
    List: Direct child metadata or names.
''')

_add_fs_chinese('GoogleDriveFS.info', '''\
使用 Google Drive 官方 files.get API 获取文件或文件夹元数据。空路径返回合成的根目录条目；其他路径使用末段文件 ID 查询。

Args:
    path (str): 文件或文件夹路径，也可以直接使用对象 ID。
    **kwargs: 预留的文件系统参数。

Returns:
    Dict[str, Any]: 标准化元数据，包含名称、标题、类型、大小和修改时间等字段。
''')
_add_fs_english('GoogleDriveFS.info', '''\
Get file or folder metadata with the official Google Drive files.get API. An empty path returns a synthetic root entry; other paths use the final segment as the object id.

Args:
    path (str): File or folder path, or an object id.
    **kwargs: Reserved filesystem options.

Returns:
    Dict[str, Any]: Normalized metadata including name, title, type, size, and modification time.
''')

_add_fs_chinese('GoogleDriveFS.read', '''\
以 UTF-8 文本读取 Google Drive 文件。普通文件通过 files.get?alt=media 下载；Google Docs 导出为纯文本，Google Sheets 导出为 CSV。暂不支持的 Google Workspace 原生格式会抛出 NotImplementedError。

Args:
    path (str): 文件路径或文件 ID。

Returns:
    str: 解码后的文件文本。
''')
_add_fs_english('GoogleDriveFS.read', '''\
Read a Google Drive file as UTF-8 text. Regular files use files.get?alt=media; Google Docs export as plain text and Google Sheets as CSV. Unsupported native Google Workspace formats raise NotImplementedError.

Args:
    path (str): File path or file id.

Returns:
    str: Decoded file text.
''')

_add_fs_chinese('GoogleDriveFS.read_file', '''\
读取完整 Google Drive 文件并返回 UTF-8 文本；下载与 Google Workspace 导出规则和 read 相同。

Args:
    path (str): 文件路径或文件 ID。

Returns:
    str: 完整文件文本。
''')
_add_fs_english('GoogleDriveFS.read_file', '''\
Read a complete Google Drive file as UTF-8 text, using the same download and Google Workspace export behavior as read.

Args:
    path (str): File path or file id.

Returns:
    str: Complete file text.
''')

_add_fs_chinese('GoogleDriveFS.write', '''\
向 Google Drive 写入 UTF-8 文本内容。该操作使用官方上传 API，需要可写 OAuth scope 或具有编辑权限的 Service Account；drive.readonly 凭据会被 Google 拒绝。

Args:
    path (str): 目标文件路径。
    content (str): 要写入的文本内容。
''')
_add_fs_english('GoogleDriveFS.write', '''\
Write UTF-8 text content to Google Drive through the official upload API. This requires a writable OAuth scope or a Service Account with edit access; drive.readonly credentials are rejected by Google.

Args:
    path (str): Destination file path.
    content (str): Text content to write.
''')

_add_fs_chinese('GoogleDriveFS.rm', '''\
删除 Google Drive 文件或文件夹。文件删除使用官方 files.delete API；recursive=True 时由统一文件系统逻辑递归处理目录。需要可写 OAuth scope，drive.readonly 凭据无法执行删除。

Args:
    path (str): 文件或文件夹路径。
    recursive (bool): 是否递归删除目录内容。默认 False。
''')
_add_fs_english('GoogleDriveFS.rm', '''\
Remove a Google Drive file or folder. File deletion uses the official files.delete API; recursive=True delegates directory traversal to the shared filesystem behavior. A writable OAuth scope is required, and drive.readonly credentials cannot delete content.

Args:
    path (str): File or folder path.
    recursive (bool): Recursively remove directory contents. Defaults to False.
''')

_add_fs_chinese('GoogleDriveFS.search', '''\
使用 Google Drive 官方 files.list API 在在线原始云盘中搜索文件正文，不查询 LazyLLM 本地知识库。支持一个或多个关键词；多个关键词按 AND 组合。可通过文件名、共享盘 ID 或直接父文件夹 ID 限定范围。

Args:
    keywords (str | List[str]): 一个关键词/短语，或多个关键词/短语。
    file_name (str, optional): 精确文件名范围。
    drive_id (str, optional): 共享盘 ID；设置后使用 corpora=drive。
    folder_id (str, optional): 直接父文件夹 ID。
    limit (int, optional): 最大结果数，范围 1 到 1000，默认 20。

Returns:
    List[Dict[str, Any]]: 匹配文件的元数据，包含 title、mime_type、google_drive_path、web_url、parents 和 drive_id。
''')
_add_fs_english('GoogleDriveFS.search', '''\
Search the live source Google Drive with the official files.list API, not a local LazyLLM knowledge base. Accepts one or more keywords combined with AND, with optional exact file-name, shared-drive, and direct parent-folder scopes.

Args:
    keywords (str | List[str]): One keyword/phrase or multiple keywords/phrases.
    file_name (str, optional): Exact file-name scope.
    drive_id (str, optional): Shared-drive id; uses corpora=drive when set.
    folder_id (str, optional): Direct parent-folder id.
    limit (int, optional): Maximum results, from 1 to 1000. Defaults to 20.

Returns:
    List[Dict[str, Any]]: File metadata including title, mime_type, google_drive_path, web_url, parents, and drive_id.
''')

_add_fs_chinese('GoogleDriveFS.find', '''\
仅按 Google Drive 文件名执行 Python 正则表达式查找。Drive API 用于按共享盘或父文件夹列出候选文件，正则匹配在本地完成，不检索文件正文。

Args:
    pattern (str): 应用于完整文件名的 Python 正则表达式。
    drive_id (str, optional): 共享盘 ID。
    folder_id (str, optional): 直接父文件夹 ID。
    limit (int, optional): 最大匹配数，默认 50。
    max_scan (int, optional): 最多检查的候选文件数，默认 1000，最大 10000。

Returns:
    List[Dict[str, Any]]: 文件名匹配的 Google Drive 文件元数据。
''')
_add_fs_english('GoogleDriveFS.find', '''\
Find Google Drive files by applying a Python regular expression to file names only. The Drive API lists candidates within optional shared-drive or parent-folder scopes; file content is not searched.

Args:
    pattern (str): Python regular expression applied to the full file name.
    drive_id (str, optional): Shared-drive id.
    folder_id (str, optional): Direct parent-folder id.
    limit (int, optional): Maximum matches. Defaults to 50.
    max_scan (int, optional): Maximum candidate files inspected. Defaults to 1000, capped at 10000.

Returns:
    List[Dict[str, Any]]: Google Drive file metadata whose names match the pattern.
''')

# OneDriveFS
_add_fs_chinese('OneDriveFS', '''\
OneDrive 文件系统：基于 Microsoft Graph API，支持 ls、读写、mkdir、rm。

构造参数:
    client_id (str, optional): Azure AD 应用的 client_id，用于应用身份模式。
    client_secret (str, optional): Azure AD 应用的 client_secret。
    tenant_id (str, optional): 租户 ID，默认 'common'。
    base_url (str, optional): Graph API 根地址，默认官方地址。

认证与配置: 推荐使用应用身份（client_id + client_secret + tenant_id，默认 'common'）由内部换取并刷新 access_token。可选 base_url。
配置与环境变量: config 项 onedrive_client_id、onedrive_client_secret、onedrive_tenant_id（环境变量 AZURE_CLIENT_ID、AZURE_CLIENT_SECRET、AZURE_TENANT_ID）；在 OneDriveFS 内解析。

如何获取 token / 应用凭证:
    方式 A — 直接使用 access_token（用户身份）:
    1. 在 Azure 门户 https://portal.azure.com 中，Azure Active Directory → 应用注册 → 新注册，记下应用程序(客户端) ID 和目录(租户) ID。
    2. 为应用配置 API 权限：Microsoft Graph → 委托权限，添加 Files.ReadWrite、User.Read 等；如需仅读可只加 Files.Read。
    3. 通过 OAuth 2.0 授权流程（如浏览器重定向或设备码流程）让用户登录并授权，用返回的 code 换取 access_token，将该 token 作为本 FS 的 token（会过期，需按 refresh_token 刷新后更新）。
    方式 B — 应用身份（client_id + client_secret）:
    1. 同上完成应用注册与 API 权限（若用应用权限，需在「应用权限」中勾选并管理员同意）。
    2. 在应用「证书和密码」中创建客户端密码，复制「值」作为 client_secret（仅显示一次）。
    3. 构造 OneDriveFS 时传入 client_id、client_secret、tenant_id（租户 ID，多租户可用 'common'），由内部调用 Microsoft 令牌端点换取 token。
''')
_add_fs_english('OneDriveFS', '''\
OneDrive FS: Microsoft Graph API; supports ls, read/write, mkdir, rm.

Auth and config: Prefer app identity flow: provide client_id, client_secret, and tenant_id (default 'common'); the FS will obtain and refresh access_token automatically. Optional base_url.
Config and env: config keys onedrive_client_id, onedrive_client_secret, onedrive_tenant_id (env AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID); resolved inside OneDriveFS.

How to obtain token / app credentials:
    Option A — Use access_token (user identity):
    1. In Azure Portal, Azure AD → App registrations → New registration; note Application (client) ID and Directory (tenant) ID.
    2. Under API permissions, add Microsoft Graph delegated permissions (e.g. Files.ReadWrite, User.Read).
    3. Complete OAuth flow to get user consent and exchange the code for access_token; use that as token for this FS (refresh when expired).
    Option B — App identity (client_id + client_secret):
    1. Same app registration; if using application permissions, add them and grant admin consent.
    2. Under Certificates & secrets, create a client secret and copy its Value as client_secret.
    3. Pass client_id, client_secret, and tenant_id (or 'common') to OneDriveFS; the class will obtain token from Microsoft token endpoint.
''')
_add_fs_example('OneDriveFS', '''\
>>> from lazyllm.tools.fs import OneDriveFS
>>> fs = OneDriveFS(client_id='xxx', client_secret='yyy', tenant_id='common')
>>> fs.ls('/')
''')

# YuqueFS
_add_fs_chinese('YuqueFS', '''\
语雀文件系统：基于语雀 API，以知识库/文档为层级，支持 ls、读写、mkdir、rm。

认证与配置: 构造参数 token（语雀 Personal Access Token）；可选 base_url。
环境变量: YUQUE_TOKEN，非空即作为 token。

如何获取 token:
    1. 登录语雀 https://www.yuque.com。
    2. 点击右上角头像 →「个人设置」→ 左侧「Token」或「访问令牌」。
    3. 点击「生成新 Token」或「生成 Token」，按需勾选权限范围（如读、写文档与知识库），生成后复制 token 字符串（仅显示一次），作为本 FS 的 token。
''')
_add_fs_english('YuqueFS', '''\
Yuque FS: Yuque API; repo/doc hierarchy; supports ls, read/write, mkdir, rm.

Auth and config: Constructor token (Yuque Personal Access Token); optional base_url.
Env var: YUQUE_TOKEN; non-empty value used as token.

How to obtain token:
    1. Log in at https://www.yuque.com.
    2. Click your avatar → Personal settings → Token (or "访问令牌") in the sidebar.
    3. Click "Generate new Token" (or "生成 Token"), select scopes (e.g. read/write docs and repos), then copy the generated token (shown once) and use it as token for this FS.
''')
_add_fs_example('YuqueFS', '''\
>>> from lazyllm.tools.fs import YuqueFS
>>> fs = YuqueFS(token='xxx')
>>> fs.ls('/')
''')

# OnesFS
_add_fs_chinese('OnesFS', '''\
ONES 项目/wiki 文件系统：基于 ONES API，以 team/space/page 为路径，支持 ls、读写、mkdir、rm。

构造参数:
    token (str): ONES 认证 token；若形如 "user_id:token" 且未单独传 user_id，则自动拆分。
    user_id (str, optional): 用户 ID；未传且 token 含 ":" 时由 token 自动解析。
    base_url (str, optional): ONES 项目 API 根地址，默认官方云地址。

认证与配置: 构造参数 token（ONES 认证 token）；可选 user_id（若未传且 token 含 ":" 则解析为 "user_id:token"）；可选 base_url。
环境变量: ONES_TOKEN，非空即作为 token。

如何获取 token:
    1. 登录 ONES（如 https://ones.ai 或自建实例）。
    2. 进入「组织设置」或「个人设置」中与「API」/「开放平台」/「访问令牌」相关的页面（不同 ONES 版本入口可能为「团队设置」→「API Token」或「个人中心」→「Token」）。
    3. 创建或复制已有的 API Token / 个人访问令牌；若接口要求带 user_id，则在构造时单独传 user_id，或将 token 写成 "user_id:token" 形式传入。
''')
_add_fs_english('OnesFS', '''\
ONES project/wiki FS: ONES API; path as team/space/page; supports ls, read/write, mkdir, rm.

Auth and config: Constructor token (ONES auth token); optional user_id (if omitted and token contains ":", parsed as "user_id:token"); optional base_url.
Env var: ONES_TOKEN; non-empty used as token.

How to obtain token:
    1. Log in to ONES (e.g. https://ones.ai or your instance).
    2. In Organization settings or Personal settings, find the API / Open platform / Access token page (e.g. Team settings → API Token, or Profile → Token, depending on version).
    3. Create or copy an API Token or personal access token; if the API requires user_id, pass user_id separately or use token in the form "user_id:token".
''')
_add_fs_example('OnesFS', '''\
>>> from lazyllm.tools.fs import OnesFS
>>> fs = OnesFS(token='xxx', user_id='uid')
>>> fs.ls('/team_uuid')
''')

# S3FS
_add_fs_chinese('S3FS', '''\
S3 兼容对象存储文件系统：基于 boto3，支持 AWS S3 及兼容协议（如 MinIO、阿里云 OSS、腾讯云 COS 等）。
路径格式为 /bucket 或 /bucket/prefix/key。支持 ls、info、读写、mkdir（桶/前缀）、rm。目录监听用 CloudFsWatchdog；支持桶事件通知（webhook）。

Args:
    token (str): 可作 access_key 使用，默认 ''。
    base_url (str, optional): 未使用，保留兼容。
    access_key (str, optional): 访问密钥，与 AWS_ACCESS_KEY_ID 等价。
    secret_key (str, optional): 秘密密钥，与 AWS_SECRET_ACCESS_KEY 等价。
    endpoint_url (str, optional): 自定义 endpoint（如 MinIO、OSS 的 endpoint）。
    region_name (str, optional): 区域名（如 us-east-1）。
    asynchronous (bool): 是否启用异步模式（高级用法，通常无需修改）。
    use_listings_cache (bool): 是否缓存目录列表（高级用法，通常无需修改）。
    skip_instance_cache (bool): 是否跳过实例缓存（高级用法，通常无需修改）。
    loop (Any, optional): 异步事件循环对象（高级用法）。

认证与配置: 构造参数 access_key、secret_key（或 token 作 access_key）；endpoint_url 用于非 AWS 兼容 endpoint；region_name 用于 AWS/区域。
环境变量（CloudFS 解析 token 用）: AWS_ACCESS_KEY_ID、S3_ACCESS_KEY、S3_TOKEN；secret 建议用 AWS_SECRET_ACCESS_KEY 或构造传入，避免写进 env。

如何获取 access_key / secret_key:
    AWS S3:
    1. 登录 AWS 控制台 https://console.aws.amazon.com，进入 IAM → 用户 → 创建用户或选择已有用户。
    2. 在「安全凭证」中点击「创建访问密钥」，选择用途（如 CLI/其他），创建后下载或复制 Access key ID 与 Secret access key（Secret 仅显示一次）。Access key ID 即 access_key，Secret access key 即 secret_key；region_name 选目标桶所在区域（如 us-east-1）。
    MinIO:
    1. 部署 MinIO 后，默认控制台可创建 Access Key 与 Secret Key；或通过 mc admin 等工具创建子账号并生成密钥。将 MinIO 服务地址（如 http://minio:9000）作为 endpoint_url 传入。
    阿里云 OSS、腾讯云 COS、其他 S3 兼容:
    1. 在对应云控制台的「访问控制」/「API 密钥」/「对象存储」相关页面创建 AccessKey ID 与 AccessKey Secret（或类似名称），并确认对象存储的 endpoint 与 region；将 endpoint 与 region 通过 endpoint_url、region_name 传入（若 SDK 要求）。
''')
_add_fs_english('S3FS', '''\
S3-compatible object storage FS: based on boto3; supports AWS S3 and compatible APIs (e.g. MinIO, Aliyun OSS, Tencent COS).
Path format: /bucket or /bucket/prefix/key. Supports ls, info, read/write, mkdir (bucket/prefix), rm. Use CloudFsWatchdog for watching; bucket event notification (webhook) supported.

Args:
    token (str): Can be used as access_key; default ''.
    base_url (str, optional): Unused; kept for compatibility.
    access_key (str, optional): Access key; same as AWS_ACCESS_KEY_ID.
    secret_key (str, optional): Secret key; same as AWS_SECRET_ACCESS_KEY.
    endpoint_url (str, optional): Custom endpoint (e.g. MinIO, OSS endpoint).
    region_name (str, optional): Region name (e.g. us-east-1).
    asynchronous (bool): Advanced async mode flag; usually not needed.
    use_listings_cache (bool): Advanced flag to cache directory listings; usually not needed.
    skip_instance_cache (bool): Advanced flag to skip instance cache; usually not needed.
    loop (Any, optional): Event loop for async environments.

Auth and config: Constructor access_key, secret_key (or token as access_key); endpoint_url for non-AWS endpoints; region_name for AWS/region.
Env vars (for CloudFS token): AWS_ACCESS_KEY_ID, S3_ACCESS_KEY, S3_TOKEN; prefer passing secret_key in code or AWS_SECRET_ACCESS_KEY, avoid plain secret in env.

How to obtain access_key / secret_key:
    AWS S3:
    1. In AWS Console, IAM → Users → create or select a user → Security credentials → Create access key; copy Access key ID (access_key) and Secret access key (secret_key, shown once). Set region_name to the bucket region (e.g. us-east-1).
    MinIO:
    1. After deploying MinIO, create Access Key and Secret Key in the console or via mc admin; use the MinIO server URL (e.g. http://minio:9000) as endpoint_url.
    Aliyun OSS, Tencent COS, other S3-compatible:
    1. In the cloud console, create AccessKey ID and Secret (or equivalent) under access control / API keys / object storage; use the service endpoint and region as endpoint_url and region_name where required.
''')
_add_fs_example('S3FS', '''\
>>> from lazyllm.tools.fs import S3FS
>>> fs = S3FS(access_key='xxx', secret_key='yyy', endpoint_url='https://s3.amazonaws.com')
>>> fs.ls('/')
>>> fs.ls('/my-bucket/path/')
>>> with fs.open('/my-bucket/file.txt', 'rb') as f:
...     data = f.read()
''')

# ObsidianFS
_add_fs_chinese('ObsidianFS', '''\
本地 Obsidian 仓库（Vault）文件系统：将本机磁盘上的 Obsidian 笔记目录映射为 FS 接口，支持 ls、info、读写、mkdir、rm、递归删除等。路径为相对 Vault 根目录的逻辑路径（如 Daily/note.md）。不依赖网络；token 填 Vault 根目录的绝对或相对路径。

Args:
    token (str): Vault 根目录路径；可为绝对路径或相对路径（相对当前工作目录），默认 '.' 表示当前目录。
    base_url (str, optional): 未使用，保留兼容。
    asynchronous (bool): 是否启用异步模式（高级用法，通常无需修改）。
    use_listings_cache (bool): 是否缓存目录列表（高级用法，通常无需修改）。
    skip_instance_cache (bool): 是否跳过实例缓存（高级用法，通常无需修改）。
    loop (Any, optional): 异步事件循环对象（高级用法）。

认证与配置: 无需 API 认证；构造时传入 token 作为 Vault 路径即可。若路径不是已存在的目录，_setup_auth 会抛出 FileNotFoundError。
环境变量（CloudFS 选用 obsidian 时）: OBSIDIAN_VAULT_PATH、OBSIDIAN_VAULT；非空值作为 token（Vault 路径）使用。

使用说明:
    1. 确保 Vault 路径存在且为目录（可在 Obsidian 中打开该仓库，复制其路径）。
    2. 通过 CloudFS(platform='obsidian', token='/path/to/vault') 或设置 OBSIDIAN_VAULT_PATH 后 CloudFS(platform='obsidian') 使用。
''')
_add_fs_english('ObsidianFS', '''\
Local Obsidian vault filesystem: maps an Obsidian vault directory on disk to the FS interface; supports ls, info, read/write, mkdir, rm, and recursive delete. Paths are logical paths relative to the vault root (e.g. Daily/note.md). No network; token is the vault root path (absolute or relative).

Args:
    token (str): Vault root directory path; absolute or relative to cwd; default '.' for current directory.
    base_url (str, optional): Unused; kept for compatibility.
    asynchronous (bool): Advanced async mode flag; usually not needed.
    use_listings_cache (bool): Advanced flag to cache directory listings; usually not needed.
    skip_instance_cache (bool): Advanced flag to skip instance cache; usually not needed.
    loop (Any, optional): Event loop for async environments.

Auth and config: No API auth; pass token as vault path. If the path is not an existing directory, _setup_auth raises FileNotFoundError.
Env vars (when CloudFS uses obsidian): OBSIDIAN_VAULT_PATH, OBSIDIAN_VAULT; non-empty value used as token (vault path).

Usage:
    1. Ensure the vault path exists and is a directory (e.g. copy path from Obsidian).
    2. Use CloudFS(platform='obsidian', token='/path/to/vault') or set OBSIDIAN_VAULT_PATH and call CloudFS(platform='obsidian').
''')
_add_fs_example('ObsidianFS', '''\
>>> from lazyllm.tools.fs import ObsidianFS
>>> fs = ObsidianFS(token='/home/user/my-vault')
>>> fs.ls('/')
>>> fs.ls('Daily')
>>> with fs.open('Daily/note.md', 'rb') as f:
...     content = f.read()
>>> fs.get_file('Daily/note.md', '/tmp/note.md')
''')
