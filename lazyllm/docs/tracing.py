# flake8: noqa E501
from . import utils
import functools
import lazyllm
import lazyllm.tracing.consume.configs
import lazyllm.tracing.consume.datamodel.raw
import lazyllm.tracing.consume.datamodel.structured
import lazyllm.tracing.consume.errors

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

该上下文以字典形式存放在 LazyLLM 全局存储的 ``trace`` 键下。``MemoryGlobals`` 按当前
操作系统线程（或 asyncio task）划分 session，因此**同一进程内并发线程默认各有一份**
``trace``；父子线程之间不会自动共享，并行 Flow 的 worker 等场景由框架在子线程里通过
``globals._update`` 显式拷入父侧数据后再执行。

Returns:
    LazyTraceContext: 当前的轻量级 tracing 上下文。
''')

add_english_doc_trace('get_trace_context', '''\
Get the current lightweight tracing context as a ``LazyTraceContext``.

The context is stored under the ``trace`` key in LazyLLM's global store. ``MemoryGlobals``
scopes storage by OS thread (or asyncio task), so **concurrent threads normally each have
their own** ``trace`` mapping; contexts are not implicitly shared across threads. Framework
code (e.g. ``Parallel`` workers) may copy parent session fields via ``globals._update`` when
a child thread must continue the same logical request.

Returns:
    LazyTraceContext: The current lightweight tracing context.
''')

add_chinese_doc_trace('set_trace_context', '''\
设置当前轻量级 tracing 上下文。

可传入 ``LazyTraceContext`` 实例或 ``dict``。字典会经 ``LazyTraceContext.from_dict`` 规范化；
其它类型会按空字典处理后再规范化。结果写入**当前 session** 的全局 ``trace`` 字段，供
runtime 与 hooks 使用。

Args:
    ctx: ``LazyTraceContext`` 实例或 tracing 上下文字典。

Returns:
    LazyTraceContext: 规范化后的 tracing 上下文对象。
''')

add_english_doc_trace('set_trace_context', '''\
Set the current lightweight tracing context.

Accepts a ``LazyTraceContext`` instance or a ``dict`` (normalized via ``LazyTraceContext.from_dict``).
Any other type is treated like an empty mapping before normalization. The result is written to
the ``trace`` field of the **current globals session** for the runtime and hooks.

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

该对象用于在全局 ``trace`` 字典中保存可序列化的 tracing 信息，例如 ``trace_id``、
``parent_span_id``、``session_id``、``user_id``、``request_tags``、运行时 trace 控制选项，
以及 ``actual_iterations``（``Loop`` 结束时按 flow id 写入的**实际循环次数**；导出
Loop span 时会读取并映射到 ``lazyllm.loop.actual_iterations``，**不会**从该 dict 中
删除条目，便于同一会话内调试与并发场景下的可观测性）。

它不承载 ``LazyTrace`` 级别的重型内存聚合；跨线程延续请求时需由调用方或框架显式
拷贝/重建上下文。
''')

add_english_doc_ctx('LazyTraceContext', '''\
Lightweight tracing context object.

This object holds serializable tracing fields for the global ``trace`` mapping, including
``trace_id``, ``parent_span_id``, ``session_id``, ``user_id``, ``request_tags``, runtime trace
flags, and ``actual_iterations`` (a map from ``Loop`` flow id to executed iteration count, written when
a loop finishes under an active trace dict; read when exporting Loop span attributes as
``lazyllm.loop.actual_iterations`` without removing entries from the dict).

It does not carry the heavy in-memory aggregate represented by ``LazyTrace``; propagating the
same logical request across threads requires explicit copy/rebuild of the context.
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

该方法会过滤未知字段，并对 ``request_tags``、``actual_iterations`` 做规范化处理
（后者若非 dict 则置为空 dict）。

Args:
    data (dict): 原始 tracing 上下文字典。

Returns:
    LazyTraceContext: 规范化后的 tracing 上下文对象。
''')

add_english_doc_ctx('LazyTraceContext.from_dict', '''\
Construct a ``LazyTraceContext`` from a dictionary.

The method filters unknown fields and normalizes ``request_tags`` and ``actual_iterations``
(the latter becomes an empty dict when the incoming value is not a dict).

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
``output_attrs``、``usage`` 等。其中 ``semantic_type`` 由 tracing runtime 根据目标对象的
``type`` / 类名 / 模块等信息集中解析（``resolve_semantic_type_for_target``），业务侧一般
无需自行赋值。``start_span`` 在成功时返回该类型的实例；``finish_span`` 消费其中的状态
完成上报。
''')

add_english_doc_span('LazySpan', '''\
A ``dataclass`` snapshot for one span, linked to the underlying OpenTelemetry span and
``LazyTrace`` aggregation state.

Fields cover entity identity (``name``, ``span_kind``, ``semantic_type``, ``component_*``),
chain identifiers (``trace_id`` / ``span_id`` / ``parent_span_id``), optional I/O and
``config``, ``output_attrs``, ``usage``, and more. ``semantic_type`` is filled by the tracing
runtime via ``resolve_semantic_type_for_target`` (from the target's ``type``, class name,
module, etc.); application code typically does not set it manually. ``start_span`` returns an
instance on success; ``finish_span`` consumes it to export telemetry.
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

# ============= Semantics (ids and semantic_type)

add_chinese_doc_sem = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.semantics)
add_english_doc_sem = functools.partial(utils.add_english_doc, module=lazyllm.tracing.semantics)

add_chinese_doc_sem('SemanticType', '''\
LazyLLM 写入 OTel 属性 ``lazyllm.semantic_type`` 时使用的取值常量（如 ``llm``、``retriever``、
``workflow_control`` 等）。具体取值由 runtime 在创建 span 时解析目标对象后选择，与
``SemanticType`` 常量集合一致。
''')

add_english_doc_sem('SemanticType', '''\
String constants used when the runtime sets the ``lazyllm.semantic_type`` OTel attribute
(e.g. ``llm``, ``retriever``, ``workflow_control``). The runtime picks a value after inspecting
the traced target; these constants describe the supported vocabulary.
''')

add_chinese_doc_sem('is_valid_trace_id', '''\
判断字符串是否为 32 位小写十六进制 W3C trace id 格式。

Args:
    value: 待校验字符串。

Returns:
    bool: 格式合法返回 ``True``。
''')

add_english_doc_sem('is_valid_trace_id', '''\
Return whether *value* matches the 32-char lowercase hex W3C trace id format.

Args:
    value: Candidate string.

Returns:
    bool: ``True`` if the format is valid.
''')

add_chinese_doc_sem('is_valid_span_id', '''\
判断字符串是否为 16 位小写十六进制 W3C span id 格式。

Args:
    value: 待校验字符串。

Returns:
    bool: 格式合法返回 ``True``。
''')

add_english_doc_sem('is_valid_span_id', '''\
Return whether *value* matches the 16-char lowercase hex W3C span id format.

Args:
    value: Candidate string.

Returns:
    bool: ``True`` if the format is valid.
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

add_chinese_doc_backend('get_consume_backend', '''\
根据名称获取 tracing 消费后端实例。

该函数会按需实例化消费后端，并复用已创建的 singleton 实例。若目标后端因依赖缺失或
导入失败而不可用，错误信息会包含可用 backend 与导入失败原因，便于定位配置或依赖问题。

Args:
    name (str): 消费 backend 名称。

Returns:
    ConsumeBackend: 对应名称的消费 backend 实例。
''')

add_english_doc_backend('get_consume_backend', '''\
Get a tracing consume backend instance by name.

The function instantiates the consume backend on demand and reuses an existing singleton
instance. If the requested backend is unavailable because an import failed, the error
message includes available backends and the import failure reason.

Args:
    name (str): The consume backend name.

Returns:
    ConsumeBackend: The consume backend instance for the given name.
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

add_chinese_doc_backend_base('ConsumeBackend', '''\
tracing 消费后端的抽象基类，定义从外部可观测系统读取 trace 原始数据的接口。

消费后端负责把后端平台的 trace / span 数据转换为 ``RawTracePayload``。上层 API 会基于
这些原始记录重建执行树并生成 ``StructuredTrace``。

**注意**: 此类是抽象基类，不能直接实例化。
''')

add_english_doc_backend_base('ConsumeBackend', '''\
Abstract base class for tracing consume backends.

A consume backend reads trace data from an external observability system and converts the
backend-specific payload into ``RawTracePayload``. Higher-level consume APIs rebuild the
execution tree from those raw records and return ``StructuredTrace``.

**Note**: This class is an abstract base class and cannot be instantiated directly.
''')

add_chinese_doc_backend_base('ConsumeBackend.fetch_trace_payload', '''\
读取完整 trace 原始载荷。

Args:
    trace_id (str): trace ID。
    timeout_seconds (float, optional): 本次读取请求的超时时间，单位秒。未传时由具体 backend
        使用自身默认值。

Returns:
    RawTracePayload: 包含 trace 记录和 span 记录列表的原始载荷。
''')

add_english_doc_backend_base('ConsumeBackend.fetch_trace_payload', '''\
Fetch the complete raw trace payload.

Args:
    trace_id (str): The trace ID.
    timeout_seconds (float, optional): Request timeout in seconds for this fetch. If omitted,
        the concrete backend uses its own default.

Returns:
    RawTracePayload: The raw payload containing the trace record and span records.
''')

add_chinese_doc_backend_base('ConsumeBackend.fetch_spans', '''\
读取指定 trace 的原始 span 记录列表。

默认实现会调用 ``fetch_trace_payload`` 并返回其中的 ``spans``。具体 backend 可以覆写该方法，
以使用更高效的后端查询接口。

Args:
    trace_id (str): trace ID。
    timeout_seconds (float, optional): 本次读取请求的超时时间，单位秒。

Returns:
    List[RawSpanRecord]: 原始 span 记录列表。
''')

add_english_doc_backend_base('ConsumeBackend.fetch_spans', '''\
Fetch raw span records for a trace.

The default implementation calls ``fetch_trace_payload`` and returns its ``spans`` field.
Concrete backends may override this method to use a more efficient backend API.

Args:
    trace_id (str): The trace ID.
    timeout_seconds (float, optional): Request timeout in seconds for this fetch.

Returns:
    List[RawSpanRecord]: Raw span records.
''')

add_chinese_doc_backend_base('ConsumeBackend.fetch_trace', '''\
读取指定 trace 的原始 trace 记录。

默认实现会调用 ``fetch_trace_payload`` 并返回其中的 ``trace``。

Args:
    trace_id (str): trace ID。
    timeout_seconds (float, optional): 本次读取请求的超时时间，单位秒。

Returns:
    RawTraceRecord: 原始 trace 记录。
''')

add_english_doc_backend_base('ConsumeBackend.fetch_trace', '''\
Fetch the raw trace record for a trace.

The default implementation calls ``fetch_trace_payload`` and returns its ``trace`` field.

Args:
    trace_id (str): The trace ID.
    timeout_seconds (float, optional): Request timeout in seconds for this fetch.

Returns:
    RawTraceRecord: The raw trace record.
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

add_chinese_doc_lf('LangfuseConsumeBackend', '''\
Langfuse tracing 消费后端实现，通过 Langfuse Public API 读取 trace 与 observation 数据。

该 backend 会将 Langfuse trace 转换为 ``RawTraceRecord``，将 observations 转换为
``RawSpanRecord``，并返回 ``RawTracePayload`` 供消费链路重建执行树。连接配置与
``LangfuseBackend`` 相同：

- ``LANGFUSE_HOST`` 或 ``LANGFUSE_BASE_URL``: Langfuse 服务地址
- ``LANGFUSE_PUBLIC_KEY``: Langfuse 公钥
- ``LANGFUSE_SECRET_KEY``: Langfuse 密钥
''')

add_english_doc_lf('LangfuseConsumeBackend', '''\
Langfuse tracing consume backend implementation that reads trace and observation data
through the Langfuse Public API.

The backend converts a Langfuse trace into ``RawTraceRecord``, converts observations into
``RawSpanRecord`` objects, and returns ``RawTracePayload`` for execution-tree reconstruction.
It uses the same connection configuration as ``LangfuseBackend``:

- ``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL``: Langfuse service URL
- ``LANGFUSE_PUBLIC_KEY``: Langfuse public key
- ``LANGFUSE_SECRET_KEY``: Langfuse secret key
''')

# ============= Consume configs

add_chinese_doc_consume_cfg = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.consume.configs)
add_english_doc_consume_cfg = functools.partial(utils.add_english_doc, module=lazyllm.tracing.consume.configs)

add_chinese_doc_consume_cfg('read_consume_backend_name', '''\
读取 tracing 消费链路默认 backend 名称。

该值来自 LazyLLM 配置项 ``trace_consume_backend``，可通过环境变量
``LAZYLLM_TRACE_CONSUME_BACKEND`` 覆写。消费链路 backend 配置与采集链路
``LAZYLLM_TRACE_BACKEND`` 相互独立。

Returns:
    str: 当前消费链路默认 backend 名称。
''')

add_english_doc_consume_cfg('read_consume_backend_name', '''\
Read the default tracing consume backend name.

The value comes from the LazyLLM configuration key ``trace_consume_backend`` and can be
overridden with the ``LAZYLLM_TRACE_CONSUME_BACKEND`` environment variable. Consume backend
configuration is separate from the collect-side ``LAZYLLM_TRACE_BACKEND``.

Returns:
    str: The current default consume backend name.
''')

add_chinese_doc_consume_cfg('read_consume_timeout_seconds', '''\
读取 tracing 消费链路请求超时时间。

该值来自 LazyLLM 配置项 ``trace_consume_timeout``，可通过环境变量
``LAZYLLM_TRACE_CONSUME_TIMEOUT`` 覆写。返回值会被限制为至少 1 秒。

Returns:
    float: 消费链路请求超时时间，单位秒。
''')

add_english_doc_consume_cfg('read_consume_timeout_seconds', '''\
Read the tracing consume request timeout.

The value comes from the LazyLLM configuration key ``trace_consume_timeout`` and can be
overridden with the ``LAZYLLM_TRACE_CONSUME_TIMEOUT`` environment variable. The returned
value is clamped to at least one second.

Returns:
    float: The consume request timeout in seconds.
''')

# ============= Consume API

add_chinese_doc_consume_api = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.consume)
add_english_doc_consume_api = functools.partial(utils.add_english_doc, module=lazyllm.tracing.consume)

add_chinese_doc_consume_api('get_single_trace', '''\
获取单条 trace 的结构化消费结果。

该函数是消费链路的高层入口：先通过消费 backend 读取 ``RawTracePayload``，再重建执行树，
最后返回符合消费接口结构的 ``StructuredTrace``。

Args:
    trace_id (str): trace ID。
    backend (str, optional): 消费 backend 名称。未传时读取 ``LAZYLLM_TRACE_CONSUME_BACKEND``，
        若环境变量未设置则默认使用 ``langfuse``。

Returns:
    StructuredTrace: 结构化 trace，包括 metadata 与 execution_tree。

Raises:
    ValueError: 当 ``trace_id`` 格式非法或 backend 不受支持时抛出。
    TraceNotFound: 当后端确认 trace 不存在时抛出。
    ConsumeBackendError: 当后端请求、鉴权或响应解析失败时抛出。
''')

add_english_doc_consume_api('get_single_trace', '''\
Fetch one trace as a structured consume result.

This is the high-level consume entry point. It reads ``RawTracePayload`` through the selected
consume backend, rebuilds the execution tree, and returns ``StructuredTrace``.

Args:
    trace_id (str): The trace ID.
    backend (str, optional): The consume backend name. If omitted, the value is read from
        ``LAZYLLM_TRACE_CONSUME_BACKEND`` and defaults to ``langfuse``.

Returns:
    StructuredTrace: The structured trace, including metadata and execution_tree.

Raises:
    ValueError: If ``trace_id`` is invalid or the backend is unsupported.
    TraceNotFound: If the backend confirms that the trace does not exist.
    ConsumeBackendError: If backend requests, authentication, or response parsing fails.
''')

# ============= Consume raw datamodel

add_chinese_doc_raw = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.consume.datamodel.raw)
add_english_doc_raw = functools.partial(utils.add_english_doc, module=lazyllm.tracing.consume.datamodel.raw)

add_chinese_doc_raw('RawTraceRecord', '''\
后端读取层返回的原始 trace 记录。

该对象只做跨 backend 的字段归一化，保留 trace 级信息，例如 ``trace_id``、``name``、
``session_id``、``user_id``、``tags``、``metadata``、输入输出、时间与原始后端载荷。
''')

add_english_doc_raw('RawTraceRecord', '''\
Raw trace record returned by the backend access layer.

This object normalizes trace-level fields across backends while keeping backend payload data,
including ``trace_id``, ``name``, ``session_id``, ``user_id``, ``tags``, ``metadata``,
input/output, timestamps, and the raw backend payload.
''')

add_chinese_doc_raw('RawSpanRecord', '''\
后端读取层返回的原始 span 记录。

该对象用于描述单个 observation/span 的父子关系、名称、状态、输入输出、属性、metadata、
错误信息与原始后端载荷。重建层会基于 ``span_id`` 和 ``parent_span_id`` 生成执行树。
''')

add_english_doc_raw('RawSpanRecord', '''\
Raw span record returned by the backend access layer.

This object describes a single observation/span, including parent-child IDs, name, status,
input/output, attributes, metadata, error information, and the raw backend payload. The
reconstruction layer builds the execution tree from ``span_id`` and ``parent_span_id``.
''')

add_chinese_doc_raw('RawTracePayload', '''\
单条 trace 的原始消费载荷。

该对象由一个 ``RawTraceRecord`` 和一组 ``RawSpanRecord`` 组成，是 backend 访问层与
reconstruction 层之间的边界对象。
''')

add_english_doc_raw('RawTracePayload', '''\
Raw consume payload for a single trace.

The payload contains one ``RawTraceRecord`` and a list of ``RawSpanRecord`` objects. It is the
boundary object between the backend access layer and the reconstruction layer.
''')

# ============= Consume structured datamodel

add_chinese_doc_structured = functools.partial(
    utils.add_chinese_doc, module=lazyllm.tracing.consume.datamodel.structured)
add_english_doc_structured = functools.partial(
    utils.add_english_doc, module=lazyllm.tracing.consume.datamodel.structured)

add_chinese_doc_structured('RawData', '''\
结构化执行节点中的原始输入输出。

Fields:
    input: 节点原始输入。
    output: 节点原始输出。
''')

add_english_doc_structured('RawData', '''\
Raw input/output data attached to a structured execution node.

Fields:
    input: The raw node input.
    output: The raw node output.
''')

add_chinese_doc_structured('TraceMetadata', '''\
结构化 trace 的 trace 级 metadata。

包含名称、开始时间、结束时间、延迟、状态、错误信息、tags、session/user 信息以及额外
metadata。该对象对应最终消费接口中的 ``metadata`` 字段。
''')

add_english_doc_structured('TraceMetadata', '''\
Trace-level metadata for a structured trace.

It contains name, start/end time, latency, status, error message, tags, session/user fields,
and additional metadata. This object corresponds to the final consume API ``metadata`` field.
''')

add_chinese_doc_structured('ExecutionStep', '''\
结构化执行树中的一个节点。

节点包含步骤 ID、名称、节点类型、语义类型、状态、开始/结束时间、延迟、原始输入输出、
语义抽取数据、错误信息以及子节点列表。
''')

add_english_doc_structured('ExecutionStep', '''\
A node in the structured execution tree.

Each node contains step ID, name, node type, semantic type, status, start/end time, latency,
raw input/output, extracted semantic data, error information, and child execution steps.
''')

add_chinese_doc_structured('StructuredTrace', '''\
消费链路返回的结构化 trace。

该对象是当前消费接口的最终结构，包含 ``trace_id``、trace 级 ``metadata``，以及根节点为
``ExecutionStep`` 的 ``execution_tree``。
''')

add_english_doc_structured('StructuredTrace', '''\
Structured trace returned by the consume pipeline.

This object is the final structure exposed by the current consume API. It contains
``trace_id``, trace-level ``metadata``, and an ``execution_tree`` rooted at ``ExecutionStep``.
''')

# ============= Consume errors

add_chinese_doc_consume_errors = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.consume.errors)
add_english_doc_consume_errors = functools.partial(utils.add_english_doc, module=lazyllm.tracing.consume.errors)

add_chinese_doc_consume_errors('ConsumeError', '''\
tracing 消费链路异常基类。

消费链路中更具体的异常，如 ``TraceNotFound`` 和 ``ConsumeBackendError``，均继承自该类。
''')

add_english_doc_consume_errors('ConsumeError', '''\
Base exception class for tracing consume errors.

More specific consume exceptions, such as ``TraceNotFound`` and ``ConsumeBackendError``,
inherit from this class.
''')

add_chinese_doc_consume_errors('TraceNotFound', '''\
指定 trace 不存在时抛出的异常。

Attributes:
    trace_id (str): 未找到的 trace ID。
''')

add_english_doc_consume_errors('TraceNotFound', '''\
Exception raised when a requested trace does not exist.

Attributes:
    trace_id (str): The trace ID that was not found.
''')

add_chinese_doc_consume_errors('ConsumeBackendError', '''\
消费 backend 访问或解析失败时抛出的异常。

常见原因包括连接配置缺失、认证失败、网络请求失败、后端返回非法 JSON 或后端响应结构不符合预期。
''')

add_english_doc_consume_errors('ConsumeBackendError', '''\
Exception raised when a consume backend request or response parsing fails.

Common causes include missing connection configuration, authentication failure, network
errors, invalid JSON responses, or unexpected backend response shapes.
''')
