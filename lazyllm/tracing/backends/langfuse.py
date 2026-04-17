import base64
import json
import os
from typing import Any, Dict, Optional

from lazyllm.thirdparty import opentelemetry
from .base import TracingBackend


_SEMANTIC_TO_LANGFUSE_TYPE = {
    'llm':              'generation',
    'retriever':        'retriever',
    'embedding':        'embedding',
    'tool':             'tool',
    'rerank':           'span',
    'workflow_control': 'chain',
}


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

    def map_attributes(self, otel_attrs: Dict[str, Any]) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        is_root_span = bool(otel_attrs.get('lazyllm.span.is_root'))

        # --- observation type from semantic_type ---
        semantic_type = otel_attrs.get('lazyllm.semantic_type')
        langfuse_type = _SEMANTIC_TO_LANGFUSE_TYPE.get(semantic_type)
        if langfuse_type:
            attrs['langfuse.observation.type'] = langfuse_type
        model = otel_attrs.get('gen_ai.request.model') or otel_attrs.get('lazyllm.entity.config.model')
        if semantic_type == 'llm' and model:
            attrs['gen_ai.request.model'] = str(model)

        # --- input / output ---
        if 'lazyllm.io.input' in otel_attrs:
            attrs['langfuse.observation.input'] = otel_attrs['lazyllm.io.input']
        if 'lazyllm.io.output' in otel_attrs:
            attrs['langfuse.observation.output'] = otel_attrs['lazyllm.io.output']

        # --- error ---
        if 'lazyllm.error.message' in otel_attrs:
            attrs['langfuse.observation.status_message'] = str(otel_attrs['lazyllm.error.message'])

        # --- usage ---
        for key in ('gen_ai.usage.input_tokens', 'gen_ai.usage.output_tokens',
                    'gen_ai.usage.total_tokens'):
            if key in otel_attrs:
                attrs[key] = otel_attrs[key]

        # --- trace-level context --- only on root spans
        if is_root_span:
            trace_name = otel_attrs.get('lazyllm.trace.name')
            if trace_name:
                attrs['langfuse.trace.name'] = trace_name

            if 'session.id' in otel_attrs:
                attrs['session.id'] = otel_attrs['session.id']
            if 'user.id' in otel_attrs:
                attrs['user.id'] = otel_attrs['user.id']

            tags = otel_attrs.get('lazyllm.trace.tags')
            if tags:
                attrs['langfuse.trace.tags'] = json.dumps(tags, ensure_ascii=False)

            if 'lazyllm.io.input' in otel_attrs:
                attrs['langfuse.trace.input'] = otel_attrs['lazyllm.io.input']
            if 'lazyllm.io.output' in otel_attrs:
                attrs['langfuse.trace.output'] = otel_attrs['lazyllm.io.output']

        return attrs
