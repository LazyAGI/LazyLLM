from typing import Optional

from lazyllm.tracing.semantics import is_valid_trace_id

from .datamodel.views import TraceDetailView


def get_single_trace(
    trace_id: str,
    *,
    backend: Optional[str] = None,
) -> TraceDetailView:
    '''按 trace_id 拉取一条完整 trace 并组装成 detail view（§9.1）。

    Args:
        trace_id: 32 位小写 hex；格式非法立即 raise ValueError。
        backend:  consume backend 名称；None → 读 config['trace_consume_backend']。

    Raises:
        TraceNotFound:        trace 在后端不存在。
        ConsumeBackendError:  后端拉取失败（网络 / 鉴权 / 反序列化）。

    Returns:
        TraceDetailView
    '''
    if not is_valid_trace_id(trace_id):
        raise ValueError(f'invalid trace_id: {trace_id!r}')
    raise NotImplementedError
