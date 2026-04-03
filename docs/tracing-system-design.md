# LazyLLM 观测系统（Tracing System）设计文档

## 1. 概述

LazyLLM 内置了一套基于 **OpenTelemetry** 的分布式链路追踪（Tracing）系统，用于在 LLM 应用执行过程中自动采集调用链信息，并将数据导出至外部可观测平台（当前支持 **Langfuse**）。

该系统的设计目标是：

- **零侵入**：Module 和 Flow 在初始化时自动注册追踪钩子，业务代码无需手动埋点
- **可插拔后端**：通过策略模式抽象后端，便于扩展至其他可观测平台
- **多层级控制**：支持全局配置、按模块白名单、按请求上下文三个维度控制追踪行为
- **惰性初始化**：OpenTelemetry 依赖在首次创建 Span 时才加载，未启用追踪时零开销

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       用户应用层                                  │
│   ModuleBase.__call__()  /  LazyLLMFlowsBase.__call__()         │
└────────────────┬────────────────────────────────────────────────┘
                 │ register_hooks / prepare_hooks / run_hooks
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Hook 层 (hook.py)                         │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  LazyLLMHook     │    │  resolve_default_hooks()         │   │
│  │  (ABC)           │    │  ── 判断是否为当前对象注册追踪Hook  │   │
│  └───────┬──────────┘    └──────────────────────────────────┘   │
│          │                                                      │
│  ┌───────▼──────────┐                                           │
│  │ LazyTracingHook  │  priority=0, error_mode='raise'           │
│  │  pre_hook()  ──────→ start_span()                            │
│  │  post_hook() ──────→ set_span_output()                       │
│  │  on_error()  ──────→ set_span_error()                        │
│  │  report()    ──────→ finish_span()                            │
│  └──────────────────┘                                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tracing Runtime 层 (tracing/)                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  _TracingRuntime (单例)                                   │   │
│  │  ── 惰性加载 OpenTelemetry                                │   │
│  │  ── 管理 TracerProvider / BatchSpanProcessor              │   │
│  │  ── 创建/更新/结束 Span                                    │   │
│  │  ── 读写 globals['trace'] 上下文                           │   │
│  └──────────────┬───────────────────────────────────────────┘   │
│                 │                                               │
│  ┌──────────────▼───────────────────────────────────────────┐   │
│  │  TracingBackend (ABC)          后端抽象                    │   │
│  │  ├─ build_exporter()           构建 OTLP Exporter        │   │
│  │  ├─ context_attributes()       请求上下文 → Span 属性     │   │
│  │  ├─ input_attributes()         输入 → Span 属性           │   │
│  │  ├─ output_attributes()        输出 → Span 属性           │   │
│  │  ├─ error_attributes()         异常 → Span 属性           │   │
│  │  └─ set_root_span_name()       设置根 Span 名称           │   │
│  └──────────────┬───────────────────────────────────────────┘   │
│                 │                                               │
│  ┌──────────────▼───────────────────────────────────────────┐   │
│  │  LangfuseBackend (具体实现)                               │   │
│  │  ── OTLP HTTP 导出至 Langfuse                             │   │
│  │  ── Basic Auth (public_key:secret_key)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────────────┘
                 │ OTLP/HTTP
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              外部可观测平台 (Langfuse)                            │
│              /api/public/otel/v1/traces                         │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 模块结构

```
lazyllm/
├── hook.py                        # Hook 框架 + LazyTracingHook
└── tracing/
    ├── __init__.py                 # 公共 API 导出 + 全局配置注册
    ├── runtime.py                  # 追踪运行时核心逻辑
    ├── configs.py                  # 模块级追踪白名单配置
    └── backends/
        ├── __init__.py             # 后端注册表
        ├── base.py                 # TracingBackend 抽象基类
        └── langfuse.py             # Langfuse OTLP 后端实现
```

## 4. 核心组件详解

### 4.1 Hook 框架 (`hook.py`)

Hook 是整个观测系统的**集成入口**。LazyLLM 定义了一个通用的 Hook 生命周期：

| 阶段 | 方法 | 调用时机 |
|------|------|---------|
| 准备 | `pre_hook(*args, **kwargs)` | 目标对象 `__call__` 执行**前** |
| 完成 | `post_hook(output)` | 目标对象 `__call__` 执行**后**，拿到输出 |
| 异常 | `on_error(exc)` | 目标对象执行抛出异常时 |
| 收尾 | `report()` | **始终执行**（在 `finally` 中），用于资源释放 |

#### LazyLLMHook 基类

```python
class LazyLLMHook(ABC):
    __hook_priority__ = 100     # 数字越小优先级越高
    __hook_error_mode__ = 'ignore'  # 'ignore' 或 'raise'
```

- `__hook_priority__`：控制多个 Hook 的执行顺序，`pre_hook` 按优先级正序，`post_hook`/`on_error`/`report` 按逆序
- `__hook_error_mode__`：`'raise'` 模式下 Hook 自身的异常会向上传播，`'ignore'` 模式下仅打印警告

#### LazyTracingHook

追踪专用 Hook 实现，**优先级为 0**（最高），**错误模式为 `raise`**：

```python
class LazyTracingHook(LazyLLMHook):
    __hook_priority__ = 0
    __hook_error_mode__ = 'raise'

    def pre_hook(self, *args, **kwargs):
        self._span_handle = start_span(span_kind=self._span_kind, target=self._obj, ...)

    def post_hook(self, output):
        set_span_output(self._span_handle, output)

    def on_error(self, exc):
        set_span_error(self._span_handle, exc)

    def report(self):
        finish_span(self._span_handle)
```

- `span_kind` 根据目标对象自动判断：有 `_flow_id` 属性为 `'flow'`，否则为 `'module'`

#### resolve_default_hooks(obj)

决定一个对象是否应该自动注册追踪 Hook：

```
1. globals['trace'].enabled == False 或 sampled == False → 不追踪
2. 对象是 Module → 通过 resolve_default_module_trace() 白名单判断
3. 对象是 Flow → 直接返回 [LazyTracingHook]
```

### 4.2 追踪运行时 (`tracing/runtime.py`)

#### _TracingRuntime（单例）

全局唯一的追踪运行时实例，负责：

1. **惰性加载 OpenTelemetry**：首次 `start_span` 时通过 `_ensure_runtime()` 初始化
2. **构建 OTLP Pipeline**：`TracerProvider → BatchSpanProcessor → OTLPSpanExporter`
3. **管理 Span 生命周期**：创建、设置属性、结束
4. **进程退出时优雅关闭**：通过 `atexit.register(self.shutdown)` 确保数据刷出

初始化流程：

```
_ensure_runtime()
    ├── import opentelemetry (失败 → TracingSetupError)
    ├── get_tracing_backend() → backend
    ├── backend.build_exporter() → exporter
    ├── Resource.create({'service.name': 'lazyllm'})
    ├── TracerProvider(resource) + BatchSpanProcessor(exporter)
    ├── trace_api.set_tracer_provider(provider)
    └── trace_api.get_tracer('lazyllm.tracing') → tracer
```

#### Span 创建逻辑 (`start_span`)

```
start_span(span_kind, target, args, kwargs)
    ├── get_trace_context() → trace_ctx
    ├── _trace_enabled(trace_ctx) → 是否追踪？
    │   ├── trace_ctx['enabled'] 优先
    │   ├── trace_ctx['sampled'] == False → 跳过
    │   └── 回退到 config['trace_enabled']
    ├── _capture_payload_enabled(trace_ctx) → 是否记录输入输出？
    ├── _target_name(target, span_kind) → span 名称
    ├── get_current_span() → 判断是否根 Span
    ├── _base_attributes() → 构建属性字典
    │   ├── lazyllm.span.kind / entity.name / entity.class / entity.id
    │   ├── backend.context_attributes() (session_id, user_id, tags)
    │   └── backend.input_attributes() (输入 payload)
    ├── tracer.start_as_current_span(name, attributes)
    └── 更新 globals['trace']['trace_id']
```

#### TraceSpanHandle

Span 句柄数据类，在 Hook 生命周期中传递：

```python
@dataclass
class TraceSpanHandle:
    span: Any           # OpenTelemetry Span 对象
    span_cm: Any        # start_as_current_span 返回的上下文管理器
    is_root_span: bool  # 是否是调用链根 Span
```

#### 请求追踪上下文 (`globals['trace']`)

通过 `lazyllm.common.globals['trace']` 在进程内传递追踪上下文：

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | `bool \| None` | 请求级开关，覆盖全局配置 |
| `trace_id` | `str \| None` | OpenTelemetry trace ID（首次创建 Span 后回写） |
| `session_id` | `str \| None` | 会话 ID（映射到 Langfuse session） |
| `user_id` | `str \| None` | 用户 ID（映射到 Langfuse user） |
| `request_tags` | `list[str]` | 请求标签 |
| `sampled` | `bool \| None` | 采样标记，`False` 时跳过追踪 |
| `parent_span_id` | `str \| None` | 父 Span ID |
| `debug_capture_payload` | `bool \| None` | 请求级覆盖是否记录输入输出 |

#### Payload 处理

输入/输出内容通过 `_stringify_payload()` 序列化，**限制为 8KB**，超出部分截断并附加 `...<truncated>` 标记。优先使用 `json.dumps`，失败时回退到 `repr()`。

### 4.3 模块追踪配置 (`tracing/configs.py`)

控制哪些 Module 默认启用追踪的白名单机制：

```python
DEFAULT_MODULE_TRACE_CONFIG = {
    'default': True,           # 默认所有模块都追踪
    'by_name': {               # 按模块名匹配
        'retriever': True,
        'reranker': True,
        'llm': True,
    },
    'by_class': {              # 按类名匹配（支持 MRO 继承链）
        'OnlineModule': True,
    },
}
```

匹配优先级：**`by_name` > `by_class`（沿 MRO 查找）> `default`**

可通过 `set_default_module_trace_config()` 运行时修改。

### 4.4 后端抽象 (`tracing/backends/`)

#### TracingBackend（ABC）

定义后端需实现的 6 个接口：

| 方法 | 职责 |
|------|------|
| `build_exporter()` | 构建 OpenTelemetry SpanExporter |
| `context_attributes(trace_ctx, is_root_span)` | 将追踪上下文映射为 Span 属性 |
| `input_attributes(args, kwargs, capture_payload, is_root_span)` | 将输入映射为 Span 属性 |
| `set_root_span_name(span, span_name)` | 设置根 Span 的显示名称 |
| `output_attributes(text, is_root_span)` | 将输出映射为 Span 属性 |
| `error_attributes(exc)` | 将异常映射为 Span 属性 |

#### LangfuseBackend

当前唯一的具体后端实现，通过 **OTLP/HTTP** 协议将数据发送至 Langfuse：

**认证方式**：HTTP Basic Auth，将 `LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY` 进行 Base64 编码

**OTLP 端点**：`{LANGFUSE_HOST}/api/public/otel/v1/traces`

**属性映射**：

| LazyLLM 概念 | Langfuse OTEL 属性 | 仅根 Span |
|-------------|-------------------|----------|
| 会话 ID | `session.id` | 是 |
| 用户 ID | `user.id` | 是 |
| 请求标签 | `langfuse.trace.tags` | 是 |
| Span 名称 | `langfuse.trace.name` | 是 |
| 输入 (trace) | `langfuse.trace.input` | 是 |
| 输出 (trace) | `langfuse.trace.output` | 是 |
| 输入 (observation) | `langfuse.observation.input` | 否 |
| 输出 (observation) | `langfuse.observation.output` | 否 |
| 错误信息 | `langfuse.observation.status_message` | 否 |

## 5. 配置体系

### 5.1 全局配置（环境变量 / `lazyllm.config`）

| 配置项 | 环境变量 | 默认值 | 说明 |
|-------|---------|-------|------|
| `trace_enabled` | `LAZYLLM_TRACE_ENABLED` | `True` | 全局追踪开关 |
| `trace_backend` | `LAZYLLM_TRACE_BACKEND` | `'langfuse'` | 后端名称 |
| `trace_content_enabled` | `LAZYLLM_TRACE_CONTENT_ENABLED` | `True` | 是否记录输入输出内容 |

### 5.2 Langfuse 后端配置（环境变量）

| 环境变量 | 必填 | 说明 |
|---------|-----|------|
| `LANGFUSE_HOST` 或 `LANGFUSE_BASE_URL` | 是 | Langfuse 服务地址 |
| `LANGFUSE_PUBLIC_KEY` | 是 | Langfuse 公钥 |
| `LANGFUSE_SECRET_KEY` | 是 | Langfuse 密钥 |

### 5.3 请求级上下文

通过 `set_trace_context()` 在运行时动态控制当前请求的追踪行为：

```python
from lazyllm.tracing import set_trace_context

set_trace_context({
    'enabled': True,
    'session_id': 'session-abc',
    'user_id': 'user-123',
    'request_tags': ['production', 'v2'],
    'sampled': True,
    'debug_capture_payload': True,
})
```

## 6. 追踪开关的判定逻辑

追踪是否生效由**三层配置**共同决定，优先级从高到低：

```
请求级 globals['trace']
    ├── enabled=False → 不追踪
    ├── sampled=False → 不追踪
    └── enabled=None → 查全局
全局 config['trace_enabled']
    ├── False → 不追踪
    └── True → 查模块白名单
模块白名单 (仅 Module)
    ├── by_name 匹配 → 按配置值
    ├── by_class MRO 匹配 → 按配置值
    └── default → 按默认值
```

对于 Flow 对象，只要全局/请求级启用，就会自动追踪。

## 7. Span 生命周期

一个完整的 Span 经历以下生命周期，映射到 Hook 的四个阶段：

```
Module/Flow.__call__(input)
    │
    ├── prepare_hooks()
    │   └── LazyTracingHook.pre_hook(input)
    │       └── start_span()
    │           ├── 创建 OpenTelemetry Span（自动建立父子关系）
    │           ├── 写入 lazyllm.* 属性
    │           ├── 写入 langfuse.* 属性
    │           └── 返回 TraceSpanHandle
    │
    ├── _call_impl(input) → output   ← 实际业务逻辑
    │
    ├── [成功] run_hooks('post_hook', output)
    │   └── LazyTracingHook.post_hook(output)
    │       └── set_span_output(handle, output)
    │           ├── span.set_attribute('lazyllm.status', 'ok')
    │           └── 写入 langfuse.observation.output / langfuse.trace.output
    │
    ├── [异常] run_hooks('on_error', exc)
    │   └── LazyTracingHook.on_error(exc)
    │       └── set_span_error(handle, exc)
    │           ├── span.set_status(ERROR)
    │           ├── span.set_attribute('lazyllm.status', 'error')
    │           ├── span.record_exception(exc)
    │           └── 写入 langfuse.observation.status_message
    │
    └── [finally] run_hooks('report')
        └── LazyTracingHook.report()
            └── finish_span(handle)
                └── span_cm.__exit__()   ← 结束 Span，数据进入 BatchSpanProcessor
```

## 8. Span 属性一览

每个 Span 携带以下属性：

### LazyLLM 通用属性

| 属性 | 说明 | 示例 |
|-----|------|------|
| `lazyllm.span.kind` | Span 类型 | `'module'` / `'flow'` |
| `lazyllm.entity.name` | 实体名称 | `'ChatModule'` |
| `lazyllm.entity.class` | 实体类名 | `'OnlineModule'` |
| `lazyllm.entity.id` | 实体 ID | Module ID / Flow ID |
| `lazyllm.status` | 执行状态 | `'ok'` / `'error'` |
| `lazyllm.request.trace_id` | 请求追踪 ID | 32 位十六进制字符串 |
| `lazyllm.request.parent_span_id` | 父 Span ID | |

### Langfuse 特有属性

见第 4.4 节属性映射表。

## 9. 依赖管理

追踪功能的依赖通过 `pyproject.toml` 的 extras 组管理：

```toml
[tool.poetry.extras]
tracing = [
    "langfuse",                                    # >= 3.0.0, < 4.0.0
    "opentelemetry-api",                           # >= 1.27.0, < 2.0.0
    "opentelemetry-sdk",                           # >= 1.27.0, < 2.0.0
    "opentelemetry-exporter-otlp-proto-http",      # >= 1.27.0, < 2.0.0
]
```

安装方式：

```bash
pip install lazyllm[tracing]
```

未安装 OpenTelemetry 依赖时，首次触发追踪会抛出 `TracingSetupError` 并打印一次性警告。

## 10. 设计模式总结

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| **策略模式 (Strategy)** | `TracingBackend` + `LangfuseBackend` | 后端可替换，通过注册表按名称查找 |
| **单例模式 (Singleton)** | `_TracingRuntime` | 进程全局唯一运行时，线程安全初始化 |
| **观察者/Hook 模式** | `LazyLLMHook` 框架 | Module/Flow 执行过程中自动触发 Hook 链 |
| **惰性初始化** | `_ensure_runtime()` | 仅在首次需要时加载 OpenTelemetry |
| **模板方法** | Hook 生命周期 | `prepare_hooks → pre_hook → call → post_hook/on_error → report` |
| **上下文传播** | `globals['trace']` + OTEL Context | 双层上下文：业务字段 + OTEL 父子关系 |

## 11. 扩展指南

### 添加新的追踪后端

1. 在 `lazyllm/tracing/backends/` 下创建新文件（如 `jaeger.py`）
2. 继承 `TracingBackend`，实现全部 6 个抽象方法
3. 在 `backends/__init__.py` 的 `_BACKENDS` 字典中注册
4. 用户通过 `LAZYLLM_TRACE_BACKEND=jaeger` 切换

```python
# lazyllm/tracing/backends/jaeger.py
from .base import TracingBackend

class JaegerBackend(TracingBackend):
    name = 'jaeger'

    def build_exporter(self):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint='http://localhost:4318/v1/traces')

    def context_attributes(self, trace_ctx, *, is_root_span):
        return {}  # Jaeger 不需要特殊上下文属性

    # ... 其他方法实现
```

```python
# lazyllm/tracing/backends/__init__.py
from .langfuse import LangfuseBackend
from .jaeger import JaegerBackend

_BACKENDS = {
    LangfuseBackend.name: LangfuseBackend(),
    JaegerBackend.name: JaegerBackend(),
}
```

## 12. 快速上手

### 最小配置启动

```bash
# 设置 Langfuse 连接信息
export LANGFUSE_HOST=https://cloud.langfuse.com
export LANGFUSE_PUBLIC_KEY=pk-lf-xxx
export LANGFUSE_SECRET_KEY=sk-lf-xxx

# 追踪默认启用，无需额外配置
python your_app.py
```

### 关闭追踪

```bash
export LAZYLLM_TRACE_ENABLED=false
```

### 仅追踪不记录内容

```bash
export LAZYLLM_TRACE_CONTENT_ENABLED=false
```

### 运行时按请求控制

```python
from lazyllm.tracing import set_trace_context

# 为特定请求启用追踪并附加用户信息
set_trace_context({
    'enabled': True,
    'user_id': 'user-123',
    'session_id': 'chat-session-456',
    'request_tags': ['canary', 'experiment-A'],
})
```
