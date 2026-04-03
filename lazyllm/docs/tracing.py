# flake8: noqa E501
from . import utils
import functools
import lazyllm

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

add_chinese_doc('TracingBackend.context_attributes', '''\
将追踪上下文转换为后端特定的 Span 属性。

Args:
    trace_ctx (Dict[str, Any]): 当前请求的追踪上下文，包含 ``session_id``、``user_id``、``request_tags`` 等字段。
    is_root_span (bool): 当前 Span 是否为调用链的根 Span。

Returns:
    Dict[str, Any]: 后端特定的 Span 属性字典。
''')

add_english_doc('TracingBackend.context_attributes', '''\
Convert the trace context into backend-specific span attributes.

Args:
    trace_ctx (Dict[str, Any]): The current request trace context containing fields such as ``session_id``, ``user_id``, and ``request_tags``.
    is_root_span (bool): Whether the current span is the root span of the trace.

Returns:
    Dict[str, Any]: A dictionary of backend-specific span attributes.
''')

add_chinese_doc('TracingBackend.input_attributes', '''\
将调用输入转换为后端特定的 Span 属性。

Args:
    args (tuple): 传递给目标对象的位置参数。
    kwargs (Dict[str, Any]): 传递给目标对象的关键字参数。
    capture_payload (bool): 是否记录输入 payload 内容。
    is_root_span (bool): 当前 Span 是否为调用链的根 Span。

Returns:
    Dict[str, Any]: 后端特定的输入属性字典。若 ``capture_payload`` 为 False，返回空字典。
''')

add_english_doc('TracingBackend.input_attributes', '''\
Convert call inputs into backend-specific span attributes.

Args:
    args (tuple): Positional arguments passed to the target object.
    kwargs (Dict[str, Any]): Keyword arguments passed to the target object.
    capture_payload (bool): Whether to record the input payload content.
    is_root_span (bool): Whether the current span is the root span of the trace.

Returns:
    Dict[str, Any]: A dictionary of backend-specific input attributes. Returns an empty dict when ``capture_payload`` is False.
''')

add_chinese_doc('TracingBackend.set_root_span_name', '''\
为根 Span 设置后端特定的显示名称。

Args:
    span: OpenTelemetry Span 对象。
    span_name (str): 要设置的 Span 名称。
''')

add_english_doc('TracingBackend.set_root_span_name', '''\
Set a backend-specific display name on the root span.

Args:
    span: The OpenTelemetry span object.
    span_name (str): The name to assign to the span.
''')

add_chinese_doc('TracingBackend.output_attributes', '''\
将调用输出转换为后端特定的 Span 属性。

Args:
    text (str): 序列化后的输出文本。
    is_root_span (bool): 当前 Span 是否为调用链的根 Span。

Returns:
    Dict[str, Any]: 后端特定的输出属性字典。
''')

add_english_doc('TracingBackend.output_attributes', '''\
Convert call output into backend-specific span attributes.

Args:
    text (str): The serialized output text.
    is_root_span (bool): Whether the current span is the root span of the trace.

Returns:
    Dict[str, Any]: A dictionary of backend-specific output attributes.
''')

add_chinese_doc('TracingBackend.error_attributes', '''\
将异常信息转换为后端特定的 Span 属性。

Args:
    exc (Exception): 执行过程中抛出的异常对象。

Returns:
    Dict[str, Any]: 后端特定的错误属性字典。
''')

add_english_doc('TracingBackend.error_attributes', '''\
Convert exception information into backend-specific span attributes.

Args:
    exc (Exception): The exception raised during execution.

Returns:
    Dict[str, Any]: A dictionary of backend-specific error attributes.
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
