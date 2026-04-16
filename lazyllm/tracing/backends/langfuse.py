import base64
import json
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

from lazyllm.thirdparty import opentelemetry
from .base import TracingBackend

if TYPE_CHECKING:
    from ..span import LazySpan


_SEMANTIC_TO_LANGFUSE_TYPE = {
    'llm':              'generation',
    'retriever':        'retriever',
    'embedding':        'embedding',
    'tool':             'tool',
    'rerank':           'span',
    'rewrite':          'span',
    'fusion':           'span',
    'context_builder':  'span',
    'workflow_control': 'chain',
    'custom':           'span',
}


def _stringify_payload(value: Any, *, limit: int = 8192) -> str:
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + '...<truncated>'
    return text


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

    def map_span_attributes(self, span: 'LazySpan') -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}

        # --- observation type from semantic_type ---
        langfuse_type = _SEMANTIC_TO_LANGFUSE_TYPE.get(span.semantic_type)
        if langfuse_type:
            attrs['langfuse.observation.type'] = langfuse_type
        if span.semantic_type == 'llm' and span.config.get('model'):
            attrs['gen_ai.request.model'] = str(span.config['model'])

        # --- input / output ---
        if span.capture_payload and span.input is not None:
            attrs['langfuse.observation.input'] = _stringify_payload(span.input)
        if span.capture_payload and span.output is not None:
            attrs['langfuse.observation.output'] = _stringify_payload(span.output)

        # --- error ---
        if span.error is not None:
            attrs['langfuse.observation.status_message'] = str(span.error)

        # --- usage ---
        if span.usage:
            prompt = span.usage.get('prompt_tokens')
            completion = span.usage.get('completion_tokens')
            if prompt is not None and prompt >= 0:
                attrs['gen_ai.usage.input_tokens'] = int(prompt)
            if completion is not None and completion >= 0:
                attrs['gen_ai.usage.output_tokens'] = int(completion)
            if (prompt is not None and prompt >= 0
                    and completion is not None and completion >= 0):
                attrs['gen_ai.usage.total_tokens'] = int(prompt + completion)

        # --- context (session / user / tags) --- only on root spans
        # (handled by map_root_span_attributes)

        return attrs

    def map_root_span_attributes(self, span: 'LazySpan') -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}

        attrs['langfuse.trace.name'] = span.name

        if span.session_id:
            attrs['session.id'] = span.session_id
        if span.user_id:
            attrs['user.id'] = span.user_id
        if span.request_tags:
            attrs['langfuse.trace.tags'] = json.dumps(span.request_tags, ensure_ascii=False)

        if span.capture_payload and span.input is not None:
            attrs['langfuse.trace.input'] = _stringify_payload(span.input)
        if span.capture_payload and span.output is not None:
            attrs['langfuse.trace.output'] = _stringify_payload(span.output)

        return attrs
