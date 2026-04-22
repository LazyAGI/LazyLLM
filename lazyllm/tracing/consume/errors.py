class ConsumeError(Exception):
    '''消费链路错误基类，便于上层一次性兜底。'''


class TraceNotFound(ConsumeError):
    def __init__(self, trace_id: str):
        super().__init__(f'trace not found: {trace_id}')
        self.trace_id = trace_id


class ConsumeBackendError(ConsumeError):
    '''后端 I/O 或反序列化失败。'''


__all__ = ['ConsumeBackendError', 'ConsumeError', 'TraceNotFound']
