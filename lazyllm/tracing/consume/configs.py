from lazyllm.configs import config


config.add(
    'trace_consume_backend',
    str,
    'langfuse',
    'TRACE_CONSUME_BACKEND',
    description='The tracing consume backend used by LazyLLM.',
)
config.add(
    'trace_consume_timeout',
    float,
    30.0,
    'TRACE_CONSUME_TIMEOUT',
    description='Timeout in seconds for tracing consume backend requests.',
)


def read_consume_backend_name() -> str:
    return config['trace_consume_backend']


def read_consume_timeout_seconds() -> float:
    value = config['trace_consume_timeout']
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 30.0


__all__ = ['read_consume_backend_name', 'read_consume_timeout_seconds']
