# Copyright (c) 2026 LazyAGI. All rights reserved.
# flake8: noqa E501
"""FS module docs: LazyLLMFSBase, CloudFSBufferedFile, CloudFS, CloudFsWatchdog, FeishuFS, ConfluenceFS, NotionFS, GoogleDriveFS, OneDriveFS, YuqueFS, OnesFS, S3FS."""
import importlib
import functools

from .. import utils

_add_fs_chinese = functools.partial(
    utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.fs'))
_add_fs_english = functools.partial(
    utils.add_english_doc, module=importlib.import_module('lazyllm.tools.fs'))
_add_fs_example = functools.partial(
    utils.add_example, module=importlib.import_module('lazyllm.tools.fs'))

# LazyLLMFSBase
_add_fs_chinese('LazyLLMFSBase', '''\
云文件系统统一基类，继承 fsspec.AbstractFileSystem，借助 registry 注册各平台实现。
子类需实现：_setup_auth、ls、info、_open、_download_range、_upload_data 等；可选实现 rm_file、mkdir。
目录监听逻辑由 CloudFsWatchdog 提供，FS 子类仅需提供必要的 webhook 能力（若有）。

Args:
    token (str): 认证 token，具体含义由子类决定（如 Bearer token、access_key 等）。
    base_url (str, optional): API 或服务根地址。
    **storage_options: 透传给 fsspec.AbstractFileSystem。
''')
_add_fs_english('LazyLLMFSBase', '''\
Unified cloud filesystem base; extends fsspec.AbstractFileSystem; implementations registered via registry.
Subclasses implement _setup_auth, ls, info, _open, _download_range, _upload_data; optionally rm_file, mkdir.
Directory watching is handled by CloudFsWatchdog; FS subclasses only expose webhook capabilities when supported.

Args:
    token (str): Auth token; meaning defined by subclass (e.g. Bearer token, access_key).
    base_url (str, optional): API or service base URL.
    **storage_options: Passed to fsspec.AbstractFileSystem.
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
在指定 path 注册 webhook。若 supports_webhook() 为 False 或子类未实现 _register_webhook，返回 {'mode': 'none'}。
支持 webhook 的子类应实现 _register_webhook(webhook_url, events, path)。

Args:
    path (str): 监控路径（如桶路径、目录 ID 等，由子类约定）。
    webhook_url (str): 回调 URL。
    events (list[str], optional): 事件类型列表，默认 ['*']。

Returns:
    dict: 含 mode（webhook/none）及子类返回的注册信息。
''')
_add_fs_english('LazyLLMFSBase.register_webhook', '''\
Register a webhook for the given path. Returns {'mode': 'none'} if supports_webhook() is False or subclass does not implement _register_webhook.
Subclasses that support webhook implement _register_webhook(webhook_url, events, path).

Args:
    path (str): Path to watch (e.g. bucket or dir id; semantics defined by subclass).
    webhook_url (str): Callback URL.
    events (list[str], optional): Event types; default ['*'].

Returns:
    dict: mode (webhook/none) and subclass-specific registration info.
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

# CloudFS
_add_fs_chinese('CloudFS', '''\
统一云文件系统入口：根据 platform 或 CLOUDFS_PLATFORM / 各平台 token 环境变量自动选择实现（feishu、confluence、notion、googledrive、onedrive、yuque、ones、s3）。
传入 platform 与 token（或 access_key/secret_key 等）即可获得对应 LazyLLMFSBase 实例。

Args:
    platform (str, optional): 平台名；不传则读 config 或按环境变量推断。
    token (str, optional): 认证 token；缺省时从各平台环境变量解析。
    **kwargs: 透传给具体 FS 实现（如 base_url、access_key、secret_key、endpoint_url、region_name 等）。

认证与配置:
    配置项 cloudfs_platform（环境变量 CLOUDFS_PLATFORM）可指定默认平台。
    未传 platform 时，按各平台 token 环境变量首次匹配到的平台作为默认平台；token 未传时从该平台对应环境变量读取。
''')
_add_fs_english('CloudFS', '''\
Unified cloud FS entry: selects implementation by platform or CLOUDFS_PLATFORM / per-platform token env (feishu, confluence, notion, googledrive, onedrive, yuque, ones, s3).
Pass platform and token (or access_key/secret_key etc.) to get the corresponding LazyLLMFSBase instance.

Args:
    platform (str, optional): Platform name; if None, from config or env.
    token (str, optional): Auth token; resolved from env when None.
    **kwargs: Passed to concrete FS (e.g. base_url, access_key, secret_key, endpoint_url, region_name).

Auth and config:
    Config key cloudfs_platform (env CLOUDFS_PLATFORM) sets default platform.
    When platform is omitted, first platform with a set token env var is used; token is then read from that platform's env vars.
''')
_add_fs_example('CloudFS', '''\
>>> from lazyllm.tools.fs import CloudFS
>>> fs = CloudFS(platform='s3', access_key='xxx', secret_key='yyy')
>>> fs.ls('/my-bucket')
>>> fs = CloudFS(platform='feishu', token='xxx')
>>> fs.ls('/')
''')

# FeishuFS
_add_fs_chinese('FeishuFS', '''\
飞书云盘文件系统：使用飞书开放平台 Drive API，支持 ls、读写、mkdir、rm。目录监听需配合 CloudFsWatchdog；部分能力支持 webhook。

认证与配置: 构造参数 token（飞书 tenant_access_token 或 user_access_token）；可选 base_url、app_id、app_secret。
环境变量（CloudFS 自动解析）: FEISHU_APP_TOKEN、FEISHU_TOKEN、LARK_TOKEN，任一非空即可作为 token。

如何获取 token:
    1. 打开飞书开放平台 https://open.feishu.cn，使用管理员账号登录。
    2. 进入「开发者后台」→「创建企业自建应用」，填写名称与描述并创建。
    3. 在应用详情「凭证与基础信息」中获取 App ID、App Secret。
    4. 在「权限管理」中为应用开通「云文档」相关权限（如 drive:drive、drive:drive.readonly 等），并发布版本/生效。
    5. tenant_access_token（以应用身份访问）: 调用开放平台「获取 tenant_access_token」接口，传入 app_id、app_secret 获取；或使用各语言 SDK 的 token 获取方法。将得到的 access_token 作为本 FS 的 token。
    6. user_access_token（以用户身份访问）: 需配置「安全设置」中的重定向 URL，通过 OAuth 授权流程让用户授权后，用返回的 code 换取 user_access_token，将该 token 作为本 FS 的 token。
''')
_add_fs_english('FeishuFS', '''\
Feishu (Lark) drive FS: uses Feishu Open Platform Drive API; supports ls, read/write, mkdir, rm. Use CloudFsWatchdog for watching; webhook supported where applicable.

Auth and config: Constructor token (Feishu tenant_access_token or user_access_token); optional base_url, app_id, app_secret.
Env vars (used by CloudFS): FEISHU_APP_TOKEN, FEISHU_TOKEN, LARK_TOKEN; any non-empty value is used as token.

How to obtain token:
    1. Go to https://open.feishu.cn (or https://open.larksuite.com), log in with an admin account.
    2. In Developer Console, create an enterprise app and get App ID and App Secret from Credentials.
    3. In Permissions, enable Drive-related scopes (e.g. drive:drive, drive:drive.readonly), then publish the app.
    4. Tenant access token: Call the "Get tenant_access_token" API with app_id and app_secret; or use the platform SDK. Use the returned access_token as token for this FS.
    5. User access token: Configure redirect URI in Security settings, complete OAuth flow for user consent, exchange the authorization code for user_access_token and use it as token.
''')
_add_fs_example('FeishuFS', '''\
>>> from lazyllm.tools.fs import FeishuFS
>>> fs = FeishuFS(token='xxx')
>>> fs.ls('/')
''')

# ConfluenceFS
_add_fs_chinese('ConfluenceFS', '''\
Confluence 文件系统：基于 Atlassian Confluence REST API，以 Space/Page 为目录与文件。目录监听用 CloudFsWatchdog；支持 webhook。

认证与配置: 构造参数 token（API token 或 Bearer token）；cloud=True 时可用 email+token 做 Basic 认证；cloud_id 用于云版 base_url。
环境变量: CONFLUENCE_TOKEN、ATLASSIAN_TOKEN，任一非空即可作为 token。

如何获取 token:
    1. 登录 Confluence（或 Atlassian 账号），点击头像 → Account settings（或 设置）。
    2. 在 Security 区域找到「Create and manage API tokens」或「API tokens」，点击进入。
    3. 点击「Create API token」，输入标签名后创建，复制生成的 token（仅显示一次），作为本 FS 的 token。
    4. 云版 Confluence（cloud=True）: 需同时提供登录邮箱（email）和上述 API token，用于 Basic 认证；cloud_id 可从 Confluence 云实例的 URL 或 Atlassian 的「API 开发」文档中获取（格式如 站点 id）。
''')
_add_fs_english('ConfluenceFS', '''\
Confluence FS: Atlassian Confluence REST API; Space/Page as dir/file. Use CloudFsWatchdog for watching; webhook supported.

Auth and config: Constructor token (API token or Bearer); when cloud=True, email+token for Basic auth; cloud_id for cloud base_url.
Env vars: CONFLUENCE_TOKEN, ATLASSIAN_TOKEN; any non-empty used as token.

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

认证与配置: 方式一：token（OAuth2 access_token）。方式二：credentials 为 JSON 文件路径或 Service Account 的 dict，自动换 token。
可选 base_url。环境变量: GOOGLE_DRIVE_TOKEN、GDRIVE_TOKEN，任一非空即可作为 token。

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

Auth and config: Option 1: token (OAuth2 access_token). Option 2: credentials as JSON path or SA dict; token fetched automatically.
Optional base_url. Env vars: GOOGLE_DRIVE_TOKEN, GDRIVE_TOKEN; any non-empty used as token.

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
>>> fs = GoogleDriveFS(token='xxx')
>>> fs.ls('/root')
''')

# OneDriveFS
_add_fs_chinese('OneDriveFS', '''\
OneDrive 文件系统：基于 Microsoft Graph API，支持 ls、读写、mkdir、rm。

认证与配置: 方式一：token（Microsoft Graph access_token）。方式二：client_id + client_secret + tenant_id（默认 'common'）由内部换取 token。
可选 base_url。环境变量: ONEDRIVE_TOKEN、MSGRAPH_TOKEN，任一非空即可作为 token。

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

Auth and config: Option 1: token (Microsoft Graph access_token). Option 2: client_id, client_secret, tenant_id (default 'common') to obtain token.
Optional base_url. Env vars: ONEDRIVE_TOKEN, MSGRAPH_TOKEN; any non-empty used as token.

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
>>> fs = OneDriveFS(token='xxx')
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
    **storage_options: 其他 fsspec 存储选项。

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
    **storage_options: Other fsspec storage options.

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
