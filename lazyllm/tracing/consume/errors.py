class ConsumeError(Exception):
    pass


class TraceNotFound(ConsumeError):
    def __init__(self, trace_id: str):
        super().__init__(f'trace not found: {trace_id}')
        self.trace_id = trace_id


class ConsumeBackendError(ConsumeError):
    pass


__all__ = ['ConsumeBackendError', 'ConsumeError', 'TraceNotFound']
