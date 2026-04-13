import base64
import json
import os
from typing import Any, Dict, Optional

from .. import opentelemetry
from .base import TracingBackend


_LANGFUSE_TRACE_NAME = 'langfuse.trace.name'
_LANGFUSE_TRACE_USER_ID = 'user.id'
_LANGFUSE_TRACE_SESSION_ID = 'session.id'
_LANGFUSE_TRACE_TAGS = 'langfuse.trace.tags'
_LANGFUSE_TRACE_INPUT = 'langfuse.trace.input'
_LANGFUSE_TRACE_OUTPUT = 'langfuse.trace.output'
_LANGFUSE_OBSERVATION_INPUT = 'langfuse.observation.input'
_LANGFUSE_OBSERVATION_OUTPUT = 'langfuse.observation.output'
_LANGFUSE_OBSERVATION_STATUS_MESSAGE = 'langfuse.observation.status_message'


class LangfuseBackend(TracingBackend):
    name = 'langfuse'

    def _config(self) -> Dict[str, Optional[str]]:
        return {
            'host': os.getenv('LANGFUSE_HOST') or os.getenv('LANGFUSE_BASE_URL'),
            'public_key': os.getenv('LANGFUSE_PUBLIC_KEY'),
            'secret_key': os.getenv('LANGFUSE_SECRET_KEY'),
        }

    def build_exporter(self):
        cfg = self._config()
        missing = [name for name, value in cfg.items() if not value]
        if missing:
            raise RuntimeError('Missing Langfuse tracing config: ' + ', '.join(missing))

        auth = base64.b64encode(f"{cfg['public_key']}:{cfg['secret_key']}".encode('utf-8')).decode('ascii')
        endpoint = cfg['host'].rstrip('/') + '/api/public/otel/v1/traces'

        OTLPSpanExporter = opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter

        return OTLPSpanExporter(
            endpoint=endpoint,
            headers={'Authorization': f'Basic {auth}'},
        )

    def context_attributes(self, trace_ctx: Dict[str, Any], *, is_root_span: bool) -> Dict[str, Any]:
        attrs = {}
        if is_root_span and trace_ctx.get('session_id'):
            attrs[_LANGFUSE_TRACE_SESSION_ID] = trace_ctx['session_id']
        if is_root_span and trace_ctx.get('user_id'):
            attrs[_LANGFUSE_TRACE_USER_ID] = trace_ctx['user_id']
        if is_root_span and trace_ctx.get('request_tags'):
            attrs[_LANGFUSE_TRACE_TAGS] = json.dumps(trace_ctx['request_tags'], ensure_ascii=False)
        return attrs

    def input_attributes(self, args: tuple[Any, ...], kwargs: Dict[str, Any], *,
                         capture_payload: bool, is_root_span: bool) -> Dict[str, Any]:
        if not capture_payload:
            return {}
        payload = {'args': args, 'kwargs': kwargs}
        attrs = {
            _LANGFUSE_OBSERVATION_INPUT: payload,
        }
        if is_root_span:
            attrs[_LANGFUSE_TRACE_INPUT] = payload
        return attrs

    def set_root_span_name(self, span: Any, span_name: str):
        span.set_attribute(_LANGFUSE_TRACE_NAME, span_name)

    def output_attributes(self, text: str, *, is_root_span: bool) -> Dict[str, Any]:
        attrs = {
            _LANGFUSE_OBSERVATION_OUTPUT: text,
        }
        if is_root_span:
            attrs[_LANGFUSE_TRACE_OUTPUT] = text
        return attrs

    def error_attributes(self, exc: Exception) -> Dict[str, Any]:
        return {
            _LANGFUSE_OBSERVATION_STATUS_MESSAGE: str(exc),
        }
