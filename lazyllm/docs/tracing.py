# flake8: noqa E501
from . import utils
import functools
import lazyllm

# ============= Tracing helpers

add_chinese_doc_trace = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing)
add_english_doc_trace = functools.partial(utils.add_english_doc, module=lazyllm.tracing)

add_chinese_doc_trace('resolve_tracing_hooks', '''\
解析并返回当前对象应自动注册的 tracing hooks。

该函数会根据全局 trace 配置、采样标志以及 module 默认 trace 策略，决定是否为当前
flow 或 module 注册 ``LazyTracingHook``。

Args:
    obj: 当前待初始化的 flow 或 module 对象。

Returns:
    list: 应自动注册到该对象上的 tracing hook 列表。
''')

add_english_doc_trace('resolve_tracing_hooks', '''\
Resolve and return tracing hooks that should be automatically registered on the current object.

This function decides whether to register ``LazyTracingHook`` for the current flow or
module according to the global trace configuration, sampling flag, and default module
trace policy.

Args:
    obj: The flow or module object currently being initialized.

Returns:
    list: A list of tracing hooks that should be automatically registered on that object.
''')

add_chinese_doc_trace('TracingSetupError', '''\
Tracing 初始化失败时抛出的异常类型。

当 tracing backend 配置非法、导出器构建失败，或运行时无法完成 tracing 初始化时，
会使用该异常表示 tracing setup 阶段的问题。
''')

add_english_doc_trace('TracingSetupError', '''\
Exception type raised when tracing initialization fails.

This exception is used for tracing setup problems such as invalid backend configuration,
exporter construction failure, or other runtime initialization errors.
''')

add_chinese_doc_trace('current_trace', '''\
返回当前执行上下文中绑定的 ``LazyTrace`` 对象。

该对象保存在 ``ContextVar`` 中，表示当前请求的 trace 级语义状态。若当前上下文尚未
创建 trace，则返回 ``None``。

Returns:
    Optional[LazyTrace]: 当前上下文对应的 trace 级对象；若不存在则返回 ``None``。
''')

add_english_doc_trace('current_trace', '''\
Return the ``LazyTrace`` object bound to the current execution context.

The object is stored in a ``ContextVar`` and represents the trace-level semantic state for
the current request. Returns ``None`` when no trace has been created in the current context.

Returns:
    Optional[LazyTrace]: The trace-level object for the current context, or ``None``.
''')

add_chinese_doc_trace('get_trace_context', '''\
获取当前轻量级 tracing 上下文 ``LazyTraceContext``。

该上下文存储在 ``globals['trace']`` 中，主要用于跨线程或跨进程传播 ``trace_id``、
``parent_span_id``、``session_id`` 和运行时 trace 控制信息。

Returns:
    LazyTraceContext: 当前的轻量级 tracing 上下文。
''')

add_english_doc_trace('get_trace_context', '''\
Get the current lightweight tracing context as a ``LazyTraceContext``.

The context is stored in ``globals['trace']`` and is mainly used to propagate ``trace_id``,
``parent_span_id``, ``session_id``, and runtime trace control information across execution
boundaries such as threads or processes.

Returns:
    LazyTraceContext: The current lightweight tracing context.
''')

add_chinese_doc_trace('set_trace_context', '''\
设置当前轻量级 tracing 上下文。

可传入 ``LazyTraceContext`` 实例或 ``dict``。字典会经 ``LazyTraceContext.from_dict`` 规范化；
其它类型会按空字典处理后再规范化。结果写入 ``lazyllm`` 全局存储中的 trace 字段，供 runtime
与 hooks 使用。

Args:
    ctx: ``LazyTraceContext`` 实例或 tracing 上下文字典。

Returns:
    LazyTraceContext: 规范化后的 tracing 上下文对象。
''')

add_english_doc_trace('set_trace_context', '''\
Set the current lightweight tracing context.

Accepts a ``LazyTraceContext`` instance or a ``dict`` (normalized via ``LazyTraceContext.from_dict``).
Any other type is treated like an empty mapping before normalization. The result is written to
the trace field in LazyLLM's global store for the runtime and hooks.

Args:
    ctx: A ``LazyTraceContext`` instance or a tracing context dictionary.

Returns:
    LazyTraceContext: The normalized tracing context object.
''')

add_chinese_doc_trace('tracing_available', '''\
检查当前环境中 tracing runtime 是否可用。

该函数会尝试确保 OpenTelemetry runtime 与 backend exporter 可以被正确初始化。

Returns:
    bool: 若 tracing runtime 可用则返回 ``True``，否则返回 ``False``。
''')

add_english_doc_trace('tracing_available', '''\
Check whether the tracing runtime is available in the current environment.

This function attempts to ensure that the OpenTelemetry runtime and the backend exporter can
be initialized correctly.

Returns:
    bool: ``True`` if the tracing runtime is available, otherwise ``False``.
''')

add_chinese_doc_trace('enable_trace', '''\
显式开启并配置一次 tracing 调用入口。

可用于两种场景：

1. 作为 wrapper：对一个 pipeline、module 或普通函数执行一次带 tracing 的调用
2. 作为 decorator：为普通函数声明一个默认的 tracing 入口配置

该函数会创建新的 request 级 tracing 上下文，并默认清理继承的 ``trace_id`` /
``parent_span_id``，从而创建新的 root trace。执行结束后，旧上下文会被恢复。

Args:
    func: 目标函数、flow 或 module。若省略，则返回 decorator。
    *args: 传递给目标对象的位置参数。
    **kwargs: tracing 配置与目标对象的关键字参数。支持的 tracing 专用参数包括
        ``trace_id``、``parent_span_id``、``session_id``、``user_id``、
        ``request_tags``、``module_trace`` 等。**注意**: 这些 tracing 专用参数会在
        构造上下文后被内部剥离，不会随剩余 ``kwargs`` 再传递给 ``func``。

Returns:
    Any: 目标对象执行后的返回值；若用作 decorator 工厂，则返回 decorator。
''')

add_english_doc_trace('enable_trace', '''\
Explicitly enable and configure tracing for a single call entry.

It supports two usage patterns:

1. As a wrapper: run a pipeline, module, or plain function with tracing enabled
2. As a decorator: declare a default tracing entry configuration for a plain function

The function creates a new request-level tracing context and, by default, clears inherited
``trace_id`` / ``parent_span_id`` so the call starts a new root trace. The previous context
is restored after execution finishes.

Args:
    func: The target function, flow, or module. When omitted, a decorator is returned.
    *args: Positional arguments passed to the target object.
    **kwargs: Tracing configuration and keyword arguments for the target object. Supported
        tracing-specific keys include ``trace_id``, ``parent_span_id``, ``session_id``,
        ``user_id``, ``request_tags``, and ``module_trace``. **Note**: these tracing-specific
        keys are consumed while constructing the trace context and are *not* forwarded to
        ``func`` together with the remaining ``kwargs``.

Returns:
    Any: The result of the target call, or a decorator when used in decorator-factory mode.
''')

add_chinese_doc_trace('start_span', '''\
在 tracing 已开启且 runtime 可用时，创建并进入一个新的 OpenTelemetry span，并返回对应的
``LazySpan`` 句柄。

通常由 ``LazyTracingHook`` 或扩展代码调用；一般业务代码更常用 ``enable_trace`` 与
flow/module 自动 hook。

Args:
    span_kind: ``flow`` / ``module`` / ``callable`` 之一。
    target: 被追踪的目标对象（flow、module 或可调用对象）。
    args: 调用位置参数元组。
    kwargs: 调用关键字参数字典。

Returns:
    Optional[LazySpan]: 成功时返回 ``LazySpan``；tracing 关闭、未采样或 runtime 不可用时返回
    ``None``。
''')

add_english_doc_trace('start_span', '''\
When tracing is enabled and the runtime is available, create and enter a new OpenTelemetry span
and return the corresponding ``LazySpan`` handle.

This is normally invoked by ``LazyTracingHook`` or extension code; typical application code
prefers ``enable_trace`` and automatic flow/module hooks.

Args:
    span_kind: One of ``flow``, ``module``, or ``callable``.
    target: The object being traced (flow, module, or callable).
    args: Positional arguments tuple for the call.
    kwargs: Keyword arguments dict for the call.

Returns:
    Optional[LazySpan]: A ``LazySpan`` on success; ``None`` if tracing is off, not sampled, or
    the runtime is unavailable.
''')

add_chinese_doc_trace('set_span_output', '''\
将成功执行结果写入 ``LazySpan``（在 ``capture_payload`` 开启时同时记录 output）。

Args:
    handle: ``start_span`` 返回的 ``LazySpan``，可为 ``None``（此时为 no-op）。
    output: 返回值或要记录的输出对象。
''')

add_english_doc_trace('set_span_output', '''\
Write a successful execution result onto the ``LazySpan`` (and persist ``output`` when
``capture_payload`` is enabled).

Args:
    handle: The ``LazySpan`` returned by ``start_span``, or ``None`` (no-op).
    output: The return value or output payload to record.
''')

add_chinese_doc_trace('set_span_usage', '''\
将 token 用量等信息写入 ``LazySpan``（例如 ``prompt_tokens`` / ``completion_tokens``），
在 ``finish_span`` 时映射为 OTel 用量属性。

Args:
    handle: ``LazySpan`` 句柄，可为 ``None``。
    usage: 用量字典。
''')

add_english_doc_trace('set_span_usage', '''\
Attach token usage (e.g. ``prompt_tokens`` / ``completion_tokens``) to the ``LazySpan``;
``finish_span`` maps these to OTel usage attributes.

Args:
    handle: The ``LazySpan`` handle, or ``None``.
    usage: Usage dictionary.
''')

add_chinese_doc_trace('set_span_attributes', '''\
将额外属性合并进 ``LazySpan.output_attrs``，在 ``finish_span`` 时写入底层 OTel span。

Args:
    handle: ``LazySpan`` 句柄，可为 ``None``。
    attrs: 要合并的属性字典。
''')

add_english_doc_trace('set_span_attributes', '''\
Merge extra attributes into ``LazySpan.output_attrs``; they are written to the underlying OTel
span at ``finish_span`` time.

Args:
    handle: The ``LazySpan`` handle, or ``None``.
    attrs: Attribute dict to merge.
''')

add_chinese_doc_trace('set_span_error', '''\
将 ``LazySpan`` 标记为错误并记录异常对象，供 ``finish_span`` 上报。

Args:
    handle: ``LazySpan`` 句柄，可为 ``None``。
    exc: 抛出的异常实例。
''')

add_english_doc_trace('set_span_error', '''\
Mark the ``LazySpan`` as failed and attach the exception for export in ``finish_span``.

Args:
    handle: The ``LazySpan`` handle, or ``None``.
    exc: The raised exception instance.
''')

add_chinese_doc_trace('finish_span', '''\
结束 ``LazySpan``：写入 lazyllm 通用 OTel 属性、调用 backend 的 ``map_attributes``、关闭
OTel span，并更新 ``LazyTrace`` / ``current_trace`` 生命周期。

Args:
    handle: ``start_span`` 返回的 ``LazySpan``，可为 ``None``（此时为 no-op）。
''')

add_english_doc_trace('finish_span', '''\
Finalize a ``LazySpan``: write generic LazyLLM OTel attributes, invoke the backend's
``map_attributes``, close the OTel span, and update ``LazyTrace`` / ``current_trace`` lifecycle.

Args:
    handle: The ``LazySpan`` from ``start_span``, or ``None`` (no-op).
''')

# ============= LazyTraceContext

add_chinese_doc_ctx = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.collect.context)
add_english_doc_ctx = functools.partial(utils.add_english_doc, module=lazyllm.tracing.collect.context)

add_chinese_doc_ctx('LazyTraceContext', '''\
轻量级 tracing 上下文对象。

该对象用于在 ``globals['trace']`` 中保存可序列化的 tracing 信息，例如 ``trace_id``、
``parent_span_id``、``session_id``、``user_id``、``request_tags`` 以及运行时 trace
控制选项。它适合做跨线程 / 跨进程传播，不承载重型 trace 语义聚合逻辑。
''')

add_english_doc_ctx('LazyTraceContext', '''\
Lightweight tracing context object.

This object stores serializable tracing information in ``globals['trace']``, such as
``trace_id``, ``parent_span_id``, ``session_id``, ``user_id``, ``request_tags``, and runtime
trace control options. It is intended for cross-thread / cross-process propagation rather
than heavy trace-level semantic aggregation.
''')

add_chinese_doc_ctx('LazyTraceContext.to_dict', '''\
将当前 ``LazyTraceContext`` 转换为普通字典。

Returns:
    dict: 当前 tracing 上下文的字典表示。
''')

add_english_doc_ctx('LazyTraceContext.to_dict', '''\
Convert the current ``LazyTraceContext`` into a plain dictionary.

Returns:
    dict: The dictionary representation of the current tracing context.
''')

add_chinese_doc_ctx('LazyTraceContext.from_dict', '''\
从字典构造 ``LazyTraceContext``。

该方法会过滤未知字段，并对 ``request_tags`` 做规范化处理。

Args:
    data (dict): 原始 tracing 上下文字典。

Returns:
    LazyTraceContext: 规范化后的 tracing 上下文对象。
''')

add_english_doc_ctx('LazyTraceContext.from_dict', '''\
Construct a ``LazyTraceContext`` from a dictionary.

The method filters unknown fields and normalizes ``request_tags`` before constructing the
context object.

Args:
    data (dict): The raw tracing context dictionary.

Returns:
    LazyTraceContext: The normalized tracing context object.
''')

# ============= Span records (collect.span)

add_chinese_doc_span = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.collect.span)
add_english_doc_span = functools.partial(utils.add_english_doc, module=lazyllm.tracing.collect.span)

add_chinese_doc_span('LazySpan', '''\
表示单次 span 的数据快照（``dataclass``），与底层 OpenTelemetry span 及 ``LazyTrace`` 聚合
状态关联。

字段涵盖实体标识（``name``、``span_kind``、``semantic_type``、``component_*``）、链信息
（``trace_id`` / ``span_id`` / ``parent_span_id``）、可选的输入输出与 ``config``、
``output_attrs``、``usage`` 等。``start_span`` 在成功时返回该类型的实例；``finish_span``
消费其中的状态完成上报。
''')

add_english_doc_span('LazySpan', '''\
A ``dataclass`` snapshot for one span, linked to the underlying OpenTelemetry span and
``LazyTrace`` aggregation state.

Fields cover entity identity (``name``, ``span_kind``, ``semantic_type``, ``component_*``),
chain identifiers (``trace_id`` / ``span_id`` / ``parent_span_id``), optional I/O and
``config``, ``output_attrs``, ``usage``, and more. ``start_span`` returns an instance on
success; ``finish_span`` consumes it to export telemetry.
''')

add_chinese_doc_span('LazyTrace', '''\
表示一次请求级 trace 的内存聚合对象，由 ``ContextVar``（``current_trace``）持有。

记录 ``trace_id``、根 ``span_id``、会话与用户信息、标签以及 ``metadata`` 等，并在多个
``LazySpan`` 开始/结束时维护计数与最终状态。根 span 结束时可能触发 ``finish`` 并清空
上下文。
''')

add_english_doc_span('LazyTrace', '''\
In-memory aggregate for one request-level trace, held in a ``ContextVar`` (``current_trace``).

Tracks ``trace_id``, root ``span_id``, session/user info, tags, ``metadata``, and span
lifecycle counters; a owning root span may ``finish`` the trace and clear context.
''')

# ============= Tracing configs

add_chinese_doc_cfg = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.collect.configs)
add_english_doc_cfg = functools.partial(utils.add_english_doc, module=lazyllm.tracing.collect.configs)

add_chinese_doc_cfg('DEFAULT_MODULE_TRACE_CONFIG', '''\
默认的 module tracing 配置。

该配置定义 tracing 默认是否开启，以及按 module 名称、按类名的默认策略。
''')

add_english_doc_cfg('DEFAULT_MODULE_TRACE_CONFIG', '''\
Default module tracing configuration.

This configuration defines whether tracing is enabled by default, as well as default policies
by module name and by class name.
''')

add_chinese_doc_cfg('get_default_module_trace_config', '''\
获取当前生效的默认 module tracing 配置。

Returns:
    dict: 当前默认 module tracing 配置的拷贝。
''')

add_english_doc_cfg('get_default_module_trace_config', '''\
Get the currently effective default module tracing configuration.

Returns:
    dict: A copy of the current default module tracing configuration.
''')

add_chinese_doc_cfg('set_default_module_trace_config', '''\
设置默认 module tracing 配置。

Args:
    config (dict): 新的默认 module tracing 配置。

Returns:
    dict: 设置完成后的规范化配置。
''')

add_english_doc_cfg('set_default_module_trace_config', '''\
Set the default module tracing configuration.

Args:
    config (dict): The new default module tracing configuration.

Returns:
    dict: The normalized configuration after the update.
''')

add_chinese_doc_cfg('resolve_default_module_trace', '''\
根据默认配置解析某个 module 是否默认开启 tracing。

Args:
    module_name: module 名称。
    module_class: module 的类对象。

Returns:
    bool: 若默认应开启 tracing 则返回 ``True``，否则返回 ``False``。
''')

add_english_doc_cfg('resolve_default_module_trace', '''\
Resolve whether tracing should be enabled by default for a given module.

Args:
    module_name: The module name.
    module_class: The module class object.

Returns:
    bool: ``True`` if tracing should be enabled by default, otherwise ``False``.
''')

add_chinese_doc_cfg('resolve_runtime_module_trace_disabled', '''\
解析运行时 ``module_trace`` 覆写是否应关闭某个 module 的 tracing。

该覆写是单向的 disable-only 语义：只允许显式关闭 tracing，不负责重新开启默认已关闭
的模块。

Args:
    override: 运行时 ``module_trace`` 配置。
    module_name: module 名称。
    module_class: module 的类对象。

Returns:
    bool: 若运行时应关闭该 module 的 tracing，则返回 ``True``。
''')

add_english_doc_cfg('resolve_runtime_module_trace_disabled', '''\
Resolve whether the runtime ``module_trace`` override should disable tracing for a module.

The override follows disable-only semantics: it can explicitly turn tracing off, but it does
not re-enable a module that is disabled by the default policy.

Args:
    override: The runtime ``module_trace`` configuration.
    module_name: The module name.
    module_class: The module class object.

Returns:
    bool: ``True`` if tracing should be disabled for the module at runtime.
''')

# ============= Semantics (ids and semantic_type literals)

add_chinese_doc_sem = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.semantics)
add_english_doc_sem = functools.partial(utils.add_english_doc, module=lazyllm.tracing.semantics)

add_chinese_doc_sem('SemanticType', '''\
LazyLLM 在 OTel span 上使用的 ``lazyllm.semantic_type`` 取值常量（如 ``llm``、``retriever``、
``workflow_control`` 等），与 tracing runtime 对 module/flow 的语义归类一致。
''')

add_english_doc_sem('SemanticType', '''\
Literal values used for ``lazyllm.semantic_type`` on OTel spans (e.g. ``llm``, ``retriever``,
``workflow_control``), matching how the tracing runtime classifies modules and flows.
''')

add_chinese_doc_sem('is_valid_trace_id', '''\
判断字符串是否为 32 位小写十六进制 W3C trace id 格式（与内部 ``_TRACE_ID_RE`` 一致）。

Args:
    value: 待校验字符串。

Returns:
    bool: 格式合法返回 ``True``。
''')

add_english_doc_sem('is_valid_trace_id', '''\
Return whether *value* matches the 32-char lowercase hex W3C trace id format (same rule as
``_TRACE_ID_RE``).

Args:
    value: Candidate string.

Returns:
    bool: ``True`` if the format is valid.
''')

add_chinese_doc_sem('is_valid_span_id', '''\
判断字符串是否为 16 位小写十六进制 span id 格式（与内部 ``_SPAN_ID_RE`` 一致）。

Args:
    value: 待校验字符串。

Returns:
    bool: 格式合法返回 ``True``。
''')

add_english_doc_sem('is_valid_span_id', '''\
Return whether *value* matches the 16-char lowercase hex span id format (same rule as
``_SPAN_ID_RE``).

Args:
    value: Candidate string.

Returns:
    bool: ``True`` if the format is valid.
''')

add_chinese_doc_sem('_TRACE_ID_RE', '''\
编译后的 trace id 正则（32 位 ``[0-9a-f]``），供 ``LazySpan`` / ``LazyTrace`` 校验与
``is_valid_trace_id`` 使用。
''')

add_english_doc_sem('_TRACE_ID_RE', '''\
Compiled regex for 32-char lowercase hex trace ids; used by ``LazySpan`` / ``LazyTrace``
validation and ``is_valid_trace_id``.
''')

add_chinese_doc_sem('_SPAN_ID_RE', '''\
编译后的 span id 正则（16 位 ``[0-9a-f]``），供 ``LazySpan`` / ``LazyTrace`` 校验与
``is_valid_span_id`` 使用。
''')

add_english_doc_sem('_SPAN_ID_RE', '''\
Compiled regex for 16-char lowercase hex span ids; used by ``LazySpan`` / ``LazyTrace``
validation and ``is_valid_span_id``.
''')

add_chinese_doc_sem('_VALID_SPAN_KINDS', '''\
允许的 ``LazySpan.span_kind`` 取值集合：``flow``、``module``、``callable``。
''')

add_english_doc_sem('_VALID_SPAN_KINDS', '''\
Frozen set of allowed ``LazySpan.span_kind`` values: ``flow``, ``module``, ``callable``.
''')

add_chinese_doc_sem('_VALID_SPAN_STATUS', '''\
允许的 ``LazySpan.status`` / ``LazyTrace.status`` 取值集合：``ok``、``error``。
''')

add_english_doc_sem('_VALID_SPAN_STATUS', '''\
Frozen set of allowed ``LazySpan.status`` / ``LazyTrace.status`` values: ``ok``, ``error``.
''')

# ============= Backends registry

add_chinese_doc_backend = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.backends)
add_english_doc_backend = functools.partial(utils.add_english_doc, module=lazyllm.tracing.backends)

add_chinese_doc_backend('get_tracing_backend', '''\
根据名称获取 tracing backend 实例。

该函数会按需实例化 backend，并复用已创建的 singleton 实例。

Args:
    name (str): backend 名称。

Returns:
    TracingBackend: 对应名称的 backend 实例。
''')

add_english_doc_backend('get_tracing_backend', '''\
Get a tracing backend instance by name.

The function instantiates the backend on demand and reuses an existing singleton instance
when available.

Args:
    name (str): The backend name.

Returns:
    TracingBackend: The backend instance for the given name.
''')

# ============= TracingBackend

add_chinese_doc_backend_base = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.backends.base)
add_english_doc_backend_base = functools.partial(utils.add_english_doc, module=lazyllm.tracing.backends.base)

add_chinese_doc_backend_base('TracingBackend', '''\
追踪后端的抽象基类，定义了将 LazyLLM 追踪数据导出至外部可观测平台所需的接口。

子类需要实现所有抽象方法，以适配具体的可观测后端（如 Langfuse、Jaeger 等）。

**注意**: 此类是抽象基类，不能直接实例化。
''')

add_english_doc_backend_base('TracingBackend', '''\
Abstract base class for tracing backends, defining the interface required to export
LazyLLM tracing data to external observability platforms.

Subclasses must implement all abstract methods to adapt to a specific observability
backend (e.g. Langfuse, Jaeger, etc.).

**Note**: This class is an abstract base class and cannot be instantiated directly.
''')

add_chinese_doc_backend_base('TracingBackend.build_exporter', '''\
构建并返回一个 OpenTelemetry SpanExporter 实例，用于将 Span 数据发送至目标后端。

Returns:
    opentelemetry.sdk.trace.export.SpanExporter: 配置好的 Span 导出器。

Raises:
    RuntimeError: 当必要的后端配置缺失时抛出。
''')

add_english_doc_backend_base('TracingBackend.build_exporter', '''\
Build and return an OpenTelemetry SpanExporter instance for sending span data to
the target backend.

Returns:
    opentelemetry.sdk.trace.export.SpanExporter: A configured span exporter.

Raises:
    RuntimeError: If required backend configuration is missing.
''')

add_chinese_doc_backend_base('TracingBackend.map_attributes', '''\
将 runtime 统一生成的通用 OTel 属性映射为后端专属的附加 Span 属性。

**契约说明**: 返回的字典仅应包含 backend 需要额外写入到 OTel span 的键值对；通用的 ``lazyllm.*``
属性已由 runtime 在调用本方法之前写入 span，子类不需要原样回传。返回的属性会被直接
``set_attribute`` 到底层 OTel span 上，后端若不需要额外映射可返回空字典。

Args:
    otel_attrs (Dict[str, Any]): runtime 在 ``finish_span`` 阶段构建出的通用 OTel 属性字典（只读）。

Returns:
    Dict[str, Any]: 需要附加到 OTel span 上的后端专属属性字典。
''')

add_english_doc_backend_base('TracingBackend.map_attributes', '''\
Map the generic OTel attributes built by the runtime into backend-specific additional span
attributes.

**Contract**: The returned dict must contain only the key/value pairs that the backend wants
to attach *in addition* to the generic ``lazyllm.*`` attributes. The runtime already writes
those generic attributes onto the span before invoking this method, so subclasses do not
need to echo them back. Returned attributes are written directly via ``set_attribute`` on
the underlying OTel span. Backends that do not need any extra mapping should return an
empty dict.

Args:
    otel_attrs (Dict[str, Any]): The generic OTel attributes constructed by the runtime during
        ``finish_span`` (treated as read-only).

Returns:
    Dict[str, Any]: Backend-specific span attributes that should be attached on top of the
    generic attributes.
''')

# ============= LangfuseBackend

add_chinese_doc_lf = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.backends.langfuse)
add_english_doc_lf = functools.partial(utils.add_english_doc, module=lazyllm.tracing.backends.langfuse)

add_chinese_doc_lf('LangfuseBackend', '''\
Langfuse 追踪后端实现，通过 OTLP/HTTP 协议将追踪数据导出至 Langfuse 平台。

使用 HTTP Basic Auth 认证，需要配置以下环境变量：

- ``LANGFUSE_HOST`` 或 ``LANGFUSE_BASE_URL``: Langfuse 服务地址
- ``LANGFUSE_PUBLIC_KEY``: Langfuse 公钥
- ``LANGFUSE_SECRET_KEY``: Langfuse 密钥
''')

add_english_doc_lf('LangfuseBackend', '''\
Langfuse tracing backend implementation that exports trace data to the Langfuse platform
via OTLP/HTTP protocol.

Uses HTTP Basic Auth authentication. The following environment variables must be configured:

- ``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL``: Langfuse service URL
- ``LANGFUSE_PUBLIC_KEY``: Langfuse public key
- ``LANGFUSE_SECRET_KEY``: Langfuse secret key
''')
