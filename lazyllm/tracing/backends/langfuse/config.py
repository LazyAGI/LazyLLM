import base64
import os
from typing import Dict, Optional


def read_consume_timeout_seconds() -> float:
    raw = os.getenv('LAZYLLM_TRACE_CONSUME_TIMEOUT', '30')
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 30.0


def read_langfuse_connection() -> Dict[str, Optional[str]]:
    return {
        'host': os.getenv('LANGFUSE_HOST') or os.getenv('LANGFUSE_BASE_URL'),
        'public_key': os.getenv('LANGFUSE_PUBLIC_KEY'),
        'secret_key': os.getenv('LANGFUSE_SECRET_KEY'),
    }


def build_basic_auth_header(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f'{public_key}:{secret_key}'.encode('utf-8')).decode('ascii')
    return f'Basic {token}'


def build_otlp_traces_endpoint(host: str) -> str:
    return host.rstrip('/') + '/api/public/otel/v1/traces'
