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

可传入 ``LazyTraceContext`` 实例或字典。该函数会统一做规范化并写入
``globals['trace']``，供后续 tracing runtime 与 hooks 使用。

Args:
    ctx: ``LazyTraceContext`` 实例或可转换为该类型的字典。

Returns:
    LazyTraceContext: 规范化后的 tracing 上下文对象。
''')

add_english_doc_trace('set_trace_context', '''\
Set the current lightweight tracing context.

Accepts either a ``LazyTraceContext`` instance or a dictionary. The input is normalized and
stored in ``globals['trace']`` for later use by the tracing runtime and hooks.

Args:
    ctx: A ``LazyTraceContext`` instance or a dictionary convertible to one.

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
    **kwargs: tracing 配置与目标对象的关键字参数。支持 ``trace_id``、
        ``parent_span_id``、``session_id``、``user_id``、``request_tags``、
        ``module_trace`` 等 tracing 参数。

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
        tracing keys include ``trace_id``, ``parent_span_id``, ``session_id``, ``user_id``,
        ``request_tags``, and ``module_trace``.

Returns:
    Any: The result of the target call, or a decorator when used in decorator-factory mode.
''')

# ============= LazyTraceContext

add_chinese_doc_ctx = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.context)
add_english_doc_ctx = functools.partial(utils.add_english_doc, module=lazyllm.tracing.context)

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

# ============= Tracing configs

add_chinese_doc_cfg = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.configs)
add_english_doc_cfg = functools.partial(utils.add_english_doc, module=lazyllm.tracing.configs)

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

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.tracing.backends.base)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.tracing.backends.base)

add_chinese_doc('TracingBackend', '''\
追踪后端的抽象基类，定义了将 LazyLLM 追踪数据导出至外部可观测平台所需的接口。

子类需要实现所有抽象方法，以适配具体的可观测后端（如 Langfuse、Jaeger 等）。

**注意**: 此类是抽象基类，不能直接实例化。
''')

add_english_doc('TracingBackend', '''\
Abstract base class for tracing backends, defining the interface required to export
LazyLLM tracing data to external observability platforms.

Subclasses must implement all abstract methods to adapt to a specific observability
backend (e.g. Langfuse, Jaeger, etc.).

**Note**: This class is an abstract base class and cannot be instantiated directly.
''')

add_chinese_doc('TracingBackend.build_exporter', '''\
构建并返回一个 OpenTelemetry SpanExporter 实例，用于将 Span 数据发送至目标后端。

Returns:
    opentelemetry.sdk.trace.export.SpanExporter: 配置好的 Span 导出器。

Raises:
    RuntimeError: 当必要的后端配置缺失时抛出。
''')

add_english_doc('TracingBackend.build_exporter', '''\
Build and return an OpenTelemetry SpanExporter instance for sending span data to
the target backend.

Returns:
    opentelemetry.sdk.trace.export.SpanExporter: A configured span exporter.

Raises:
    RuntimeError: If required backend configuration is missing.
''')

add_chinese_doc('TracingBackend.map_attributes', '''\
将 runtime 统一生成的通用 OTel 属性映射为后端专属的 Span 属性。

Args:
    otel_attrs (Dict[str, Any]): runtime 在 ``finish_span`` 阶段构建出的通用 OTel 属性字典。

Returns:
    Dict[str, Any]: 后端特定的 Span 属性字典。
''')

add_english_doc('TracingBackend.map_attributes', '''\
Map the generic OTel attributes built by the runtime into backend-specific span attributes.

Args:
    otel_attrs (Dict[str, Any]): The generic OTel attributes constructed by the runtime during
        ``finish_span``.

Returns:
    Dict[str, Any]: A dictionary of backend-specific span attributes.
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
