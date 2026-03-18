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

# LazyLLMFSBase
_add_fs_chinese('LazyLLMFSBase', '''\
云文件系统统一基类，继承 fsspec.AbstractFileSystem，借助 registry 注册各平台实现。
子类需实现：_setup_auth、ls、info、_open、_download_range、_upload_data 等；可选实现 rm_file、mkdir。
目录监听逻辑由 CloudFsWatchdog 提供，FS 子类仅需提供必要的 webhook 能力（若有）。

Args:
    token (Any): 认证信息载体，通常为字符串 token，部分子类也可能封装为 dict/tuple 等结构。
    base_url (str, optional): API 或服务根地址。
    asynchronous (bool): 是否启用异步模式，对应 fsspec.AbstractFileSystem.asynchronous，默认 False。
    use_listings_cache (bool): 是否缓存目录列表，对应 fsspec.AbstractFileSystem.use_listings_cache，默认 False。
    skip_instance_cache (bool): 是否跳过实例缓存，对应 fsspec.AbstractFileSystem.skip_instance_cache，默认 False。
    loop (Any, optional): 异步事件循环对象，一般仅在异步环境下需要显式传入。
''')
_add_fs_english('LazyLLMFSBase', '''\
Unified cloud filesystem base; extends fsspec.AbstractFileSystem; implementations registered via registry.
Subclasses implement _setup_auth, ls, info, _open, _download_range, _upload_data; optionally rm_file, mkdir.
Directory watching is handled by CloudFsWatchdog; FS subclasses only expose webhook capabilities when supported.

Args:
    token (Any): Auth payload; usually a string token, but some subclasses may encode auth bundles (e.g. dict/tuple) here.
    base_url (str, optional): API or service base URL.
    asynchronous (bool): Whether to enable async mode; forwarded to fsspec.AbstractFileSystem.asynchronous; default False.
    use_listings_cache (bool): Whether to cache directory listings; forwarded to fsspec.AbstractFileSystem.use_listings_cache; default False.
    skip_instance_cache (bool): Whether to skip the filesystem instance cache; forwarded to fsspec.AbstractFileSystem.skip_instance_cache; default False.
    loop (Any, optional): Event loop object for async environments.
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
    list: detail=True 时为 dict 列表；detail=False 时为 name 列表。
''')
_add_fs_english('LazyLLMFSBase.ls', '''\
List directory entries at path. Must be implemented by subclass.

Args:
    path (str): Directory path.
    detail (bool): If True return list of dicts (name, size, type, mtime etc.); if False return list of names.
    **kwargs: Subclass-specific options.

Returns:
    list: List of dicts if detail=True else list of names.
''')

_add_fs_chinese('LazyLLMFSBase.info', '''\
获取路径对应条目元信息。子类必须实现。

Args:
    path (str): 文件或目录路径。
    **kwargs: 子类扩展参数。

Returns:
    dict: 至少包含 name, size, type（file/directory），可选 mtime 等。
''')
_add_fs_english('LazyLLMFSBase.info', '''\
Get metadata for the path. Must be implemented by subclass.

Args:
    path (str): File or directory path.
    **kwargs: Subclass-specific options.

Returns:
    dict: At least name, size, type (file/directory); optional mtime etc.
''')

_add_fs_chinese('LazyLLMFSBase.open', '''\
以 fsspec 方式打开文件，返回可读/可写文件对象。内部调用子类实现的 _open。

Args:
    path (str): 文件路径。
    mode (str): 如 'rb'、'wb'。
    block_size (int, optional): 缓冲块大小。
    **kwargs: 透传给 _open。

Returns:
    CloudFSBufferedFile: 支持区间读、整块写的缓冲文件对象。
''')
_add_fs_english('LazyLLMFSBase.open', '''\
Open file in fsspec style; returns readable/writable file-like object. Uses subclass _open.

Args:
    path (str): File path.
    mode (str): e.g. 'rb', 'wb'.
    block_size (int, optional): Buffer block size.
    **kwargs: Passed to _open.

Returns:
    CloudFSBufferedFile: Buffered file supporting range read and chunk write.
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
    maxdepth (int, optional): 未使用，保留兼容。
''')
_add_fs_english('LazyLLMFSBase.rm', '''\
Remove path; if directory and recursive=True, recursively delete then rmdir; else rm_file.

Args:
    path (str): File or directory path.
    recursive (bool): Whether to recursively delete directory.
    maxdepth (int, optional): Unused; kept for compatibility.
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

_add_fs_chinese('LazyLLMFSBase._ensure_token', '''\
Token 刷新钩子：子类可覆盖。基类默认实现为 no-op。
每次 _request 发请求前会调用本方法；使用会过期的 access token 的子类应覆盖本方法，在 token 即将过期或已过期时重新获取并更新 session 认证信息。
''')
_add_fs_english('LazyLLMFSBase._ensure_token', '''\
Token refresh hook; subclasses may override. Default implementation is a no-op.
Called before each _request; subclasses that use expiring access tokens should override to refresh the token and update session auth when needed.
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
_add_fs_chinese('FeishuFS', '''\
飞书云盘文件系统：使用飞书开放平台 Drive API，支持 ls、读写、mkdir、rm。目录监听需配合 CloudFsWatchdog；部分能力支持 webhook。

构造参数:
    base_url (str, optional): 飞书开放平台根地址，默认使用官方地址即可。
    app_id (str, optional): 企业自建应用的 App ID，用于换取 tenant_access_token 或刷新 user_access_token。
    app_secret (str, optional): 企业自建应用的 App Secret。
    space_id (str, optional): 若传入则会返回 FeishuWikiFS 实例，将指定知识库作为 FS。
    云盘上传：put_file 时若远程路径以 .md 结尾或传入 content_type='markdown'，会在目标目录创建 docx 并按 Markdown 解析写入（标题、列表、代码块等）；否则按二进制上传原文件。
    user_refresh_token (str, optional): OAuth2 refresh_token，用于以用户身份访问「我的空间」个人文件。
        - 传入真实 refresh_token：直接用于换取 user_access_token，每次刷新后内存中的 token 同步滚动。
        - 传入 'auto'：自动触发 OAuth2 授权流程——在本地启动临时回调服务，将授权链接通过
          lazyllm.LOG.success 输出到终端，用户在浏览器中点击并完成授权后，自动交换得到 refresh_token。
        飞书 refresh_token 有效期约 7 天，每次使用后立即作废并换发新值；只要在有效期内使用即可持续续期
        （最长 365 天后需重新 OAuth 授权）。token 的持久化由调用方负责，LazyLLM 不做本地存储。
        若不设置则使用 tenant_access_token（仅能访问已授权给应用的共享文件）。
    oauth_port (int, optional): 'auto' 流程中本地回调服务的监听端口，默认 9981。
        使用前需在飞书应用「安全设置」→「重定向 URL」中添加 http://localhost:{oauth_port}/callback。
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
_add_fs_english('FeishuFS', '''\
Feishu (Lark) drive FS: uses Feishu Open Platform Drive API; supports ls, read/write, mkdir, rm. Use CloudFsWatchdog for watching; webhook supported where applicable.

Parameters:
    base_url (str, optional): Feishu Open Platform base URL; defaults to the official endpoint.
    app_id (str, optional): App ID of the enterprise custom app, used to obtain tenant_access_token or refresh user_access_token.
    app_secret (str, optional): App Secret of the enterprise custom app.
    space_id (str, optional): When provided, returns a FeishuWikiFS instance targeting the given wiki space.
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
        http://localhost:{oauth_port}/callback must be pre-registered in the Feishu app Security Settings → Redirect URL.
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
>>> from lazyllm.tools.fs import FeishuFS
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
''')

# FeishuWikiFS
_add_fs_chinese('FeishuWikiFS', '''\
飞书知识库文件系统：使用飞书开放平台 Wiki API，将指定知识库（space）映射为文件系统。目录对应 Wiki 目录/节点，文件对应文档或附件。支持 ls、info、open（读/写）、mkdir、rm_file、put_file；并支持文档块级文本编辑：get_document_id、get_doc_blocks、update_doc_block_text。

使用方式:
    构造参数:
        base_url (str, optional): 飞书开放平台根地址，默认使用官方地址即可。
        app_id (str, optional): 企业自建应用的 App ID。
        app_secret (str, optional): 企业自建应用的 App Secret。
        space_id (str): 知识空间 ID，例如 'wikcnKQ1k3pcuo5uSK4t8VN6kVf'。

    1. 与 FeishuFS 共享同一套鉴权方式：推荐 app_id + app_secret，由 FS 自动获取 tenant_access_token。
    2. 必须传入 space_id（知识空间 ID），例如 'wikcnKQ1k3pcuo5uSK4t8VN6kVf'。
    3. 根路径 '/' 对应知识库根节点；路径为 Wiki 节点路径；open 会根据节点类型拉取文档纯文本或附件二进制；put_file 会新建 docx 并写入内容。当远程路径以 .md 结尾或 put_file(..., content_type='markdown') 时，会按 Markdown 解析并保持标题、列表、代码块、引用等格式；否则按纯文本追加段落。

保持原格式（表格、标题等）:
    - 通过 open/raw_content 下载到的是纯文本，put_file 会新建文档并只追加段落，因此「下载-修改-上传」会丢失表格等格式。
    - 若要保留原格式，请使用块级编辑：先 get_doc_blocks(path) 获取文档块列表（含 block_id、block_type、plain_text），再对需要修改的文本块调用 update_doc_block_text(path, block_id, new_text)。仅修改目标文本块，表格等其它块不会被改动。
''')
_add_fs_english('FeishuWikiFS', '''\
Feishu Wiki FS: Feishu Open Platform Wiki API; maps a wiki space into a filesystem. Directories are wiki folders/nodes; files are documents or attachments. Supports ls, info, open (r/w), mkdir, rm_file, put_file, and block-level text edit: get_document_id, get_doc_blocks, update_doc_block_text.

Usage:
    1. Shares the same auth model as FeishuFS; space_id (wiki space ID) is required.
    2. '/' is the wiki root; paths are wiki node paths; open returns document plain text or file binary; put_file creates a new docx. When the remote path ends with .md or put_file(..., content_type='markdown'), content is parsed as Markdown and rendered as headings, lists, code blocks, quotes, etc.; otherwise appended as plain text paragraphs.

Preserving format (tables, headings, etc.):
    - open/raw_content returns plain text only; put_file creates a new doc and appends paragraphs, so download-modify-upload loses tables and other structure.
    - To preserve format, use block-level edit: get_doc_blocks(path) to list blocks (block_id, block_type, plain_text), then update_doc_block_text(path, block_id, new_text) for the blocks you need to change; other blocks (e.g. tables) are left unchanged.
''')
_add_fs_chinese('FeishuWikiFS.get_document_id', '''\
返回 Wiki 文档节点对应的飞书 docx document_id（即 obj_token）。path 必须指向 doc 或 docx 类型节点，否则抛出 ValueError。

Args:
    path (str): Wiki 内文档路径（如 '/一级/文档标题'）。

Returns:
    str: 文档的 document_id，用于飞书 docx API。
''')
_add_fs_english('FeishuWikiFS.get_document_id', '''\
Return the Feishu docx document_id (obj_token) for the wiki document at path. path must be a doc or docx node, otherwise ValueError is raised.

Args:
    path (str): Wiki path to the document (e.g. '/level1/doc title').

Returns:
    str: The document_id for Feishu docx API.
''')
_add_fs_chinese('FeishuWikiFS.get_doc_blocks', '''\
获取文档的块列表（block 树扁平列表）。用于在编辑时定位要修改的块并保留表格等非文本块。

Args:
    path (str): Wiki 内文档路径。
    with_descendants (bool): 是否包含所有子孙块，默认 True。

Returns:
    list: 每项为 dict，含 block_id、block_type、parent_id；若为文本类块则含 plain_text。
''')
_add_fs_english('FeishuWikiFS.get_doc_blocks', '''\
Get the document block list (flattened block tree). Use to locate blocks to edit while leaving tables and other non-text blocks unchanged.

Args:
    path (str): Wiki path to the document.
    with_descendants (bool): Whether to include all descendant blocks; default True.

Returns:
    list: Each item is a dict with block_id, block_type, parent_id; text blocks also have plain_text.
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
_add_fs_chinese('NotionFS', '''\
Notion 文件系统：基于 Notion API，以 Page/Block 为层级，支持 ls、读写、mkdir、rm（归档）。
写入页面内容时，Notion API 单 block 的 rich_text 限制为 2000 字符，超长内容会被截断。

认证与配置: 构造参数 token（Notion Integration Token / Internal Integration Secret）；可选 base_url。
环境变量: NOTION_TOKEN、NOTION_INTEGRATION_TOKEN，任一非空即可作为 token。

如何获取 token:
    1. 登录 https://www.notion.so，进入要管理的 Workspace。
    2. 侧栏底部点击「Settings & members」→「Connections」或「Integrations」。
    3. 点击「Develop or manage integrations」或「New integration」，创建新集成（Integration）。
    4. 在集成详情页复制「Internal Integration Secret」或「API key」（形如 secret_xxx），即为本 FS 的 token。注意：需在需要访问的页面/数据库中，通过「Connections」或「Add connections」将该集成连接上，否则 API 无法访问该内容。
''')
_add_fs_english('NotionFS', '''\
Notion FS: Notion API; Page/Block hierarchy; supports ls, read/write, mkdir, rm (archive).
When writing page content, Notion API limits rich_text to 2000 chars per block; longer content is truncated.

Auth and config: Constructor token (Notion Integration Token / Internal Integration Secret); optional base_url.
Env vars: NOTION_TOKEN, NOTION_INTEGRATION_TOKEN; any non-empty used as token.

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

# GoogleDriveFS
_add_fs_chinese('GoogleDriveFS', '''\
Google Drive 文件系统：使用 Drive v3 API，支持 OAuth token 或 Service Account 凭证。目录监听用 CloudFsWatchdog；支持 webhook。

构造参数:
    credentials (str | dict, optional): Service Account JSON 文件路径或已解析的 dict；提供后会自动换取 access_token。
    base_url (str, optional): Drive API 根地址，默认官方地址。

认证与配置: 推荐使用 Service Account 方式：credentials 为 JSON 文件路径或 Service Account 的 dict，由 FS 自动换取并刷新 access_token。可选 base_url。
配置与环境变量: config 项 googledrive_credentials（环境变量 GOOGLE_APPLICATION_CREDENTIALS）指向 Service Account JSON 路径；在 GoogleDriveFS 内解析。

如何获取 token / 凭证:
    方式 A — OAuth2 access_token（用户身份）:
    1. 打开 Google Cloud Console https://console.cloud.google.com，创建或选择项目。
    2. 启用「Google Drive API」：APIs & Services → Enable APIs and Services → 搜索并启用 Drive API。
    3. 创建 OAuth 2.0 凭据：APIs & Services → Credentials → Create Credentials → OAuth client ID，应用类型选 Desktop 或 Web，获取 client_id 与 client_secret。
    4. 使用 Google 官方 OAuth 流程（或 google-auth 等库）让用户授权，用返回的 authorization code 换取 access_token 和 refresh_token；将 access_token 作为本 FS 的 token（注意 access_token 有过期时间，生产环境建议用 refresh_token 定期刷新后传入）。
    方式 B — Service Account（服务身份）:
    1. 同上在 Cloud Console 启用 Drive API。
    2. 创建服务账号：APIs & Services → Credentials → Create Credentials → Service account，创建后进入该服务账号 → Keys → Add key → JSON，下载 JSON 密钥文件。
    3. 将需要访问的 Google Drive 文件夹或「我的云端硬盘」共享给该服务账号的邮箱（如 xxx@yyy.iam.gserviceaccount.com），权限至少为「查看者」或「编辑者」。
    4. 构造 GoogleDriveFS 时传入 credentials 为该 JSON 文件路径或已解析的 dict，无需手动传 token（内部会用 JSON 中的私钥换取 access_token）。
''')
_add_fs_english('GoogleDriveFS', '''\
Google Drive FS: Drive v3 API; OAuth token or Service Account credentials. Use CloudFsWatchdog for watching; webhook supported.

Auth and config: Prefer Service Account credentials: pass credentials as JSON path or SA dict so the FS can obtain and refresh access_token automatically. Optional base_url.
Config and env: config key googledrive_credentials (env GOOGLE_APPLICATION_CREDENTIALS) points to Service Account JSON path; resolved inside GoogleDriveFS.

How to obtain token / credentials:
    Option A — OAuth2 access_token (user identity):
    1. In Google Cloud Console (https://console.cloud.google.com), create or select a project and enable the Google Drive API.
    2. Create OAuth 2.0 credentials (Desktop or Web client), get client_id and client_secret.
    3. Run the OAuth flow (e.g. with google-auth) to get user consent and exchange the authorization code for access_token (and refresh_token). Use access_token as token for this FS; refresh it with refresh_token when expired.
    Option B — Service Account:
    1. Enable Drive API and create a Service Account in the same project; download its JSON key from Keys → Add key → JSON.
    2. Share the target Drive folder (or My Drive) with the service account email (e.g. xxx@yyy.iam.gserviceaccount.com) with at least Viewer/Editor access.
    3. Pass credentials as the JSON file path or parsed dict to GoogleDriveFS; no need to pass token (the class will obtain access_token from the key).
''')
_add_fs_example('GoogleDriveFS', '''\
>>> from lazyllm.tools.fs import GoogleDriveFS
>>> fs = GoogleDriveFS(credentials='/path/to/service-account.json')
>>> fs.ls('/root')
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
