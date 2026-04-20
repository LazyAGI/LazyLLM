import base64
import json
import os
from typing import Any, Dict, Optional

from lazyllm.common import LOG
from lazyllm.thirdparty import opentelemetry
from ..semantics import SemanticType
from .base import TracingBackend


_SEMANTIC_TO_LANGFUSE_TYPE = {
    SemanticType.AGENT: 'chain',
    SemanticType.LLM: 'generation',
    SemanticType.RETRIEVER: 'retriever',
    SemanticType.EMBEDDING: 'embedding',
    SemanticType.TOOL: 'tool',
    SemanticType.RERANK: 'span',
    SemanticType.WORKFLOW_CONTROL: 'chain',
}


class LangfuseBackend(TracingBackend):
    name = 'langfuse'

    _warned_semantic_types: set = set()

    @staticmethod
    def _copy_usage_attrs(attrs: Dict[str, Any], otel_attrs: Dict[str, Any]) -> None:
        for key in (
            'gen_ai.usage.input_tokens',
            'gen_ai.usage.output_tokens',
            'gen_ai.usage.total_tokens',
        ):
            if key in otel_attrs:
                attrs[key] = otel_attrs[key]

    @staticmethod
    def _copy_trace_attrs(attrs: Dict[str, Any], otel_attrs: Dict[str, Any]) -> None:
        trace_name = otel_attrs.get('lazyllm.trace.name')
        if trace_name:
            attrs['langfuse.trace.name'] = trace_name
        if 'session.id' in otel_attrs:
            attrs['session.id'] = otel_attrs['session.id']
        if 'user.id' in otel_attrs:
            attrs['user.id'] = otel_attrs['user.id']

        tags = otel_attrs.get('lazyllm.trace.tags')
        if tags:
            try:
                attrs['langfuse.trace.tags'] = json.dumps(tags, ensure_ascii=False)
            except TypeError as exc:
                LOG.warning(f'Failed to JSON-serialize lazyllm.trace.tags ({type(tags).__name__}): {exc}; '
                            f'falling back to string coercion.')
                attrs['langfuse.trace.tags'] = json.dumps(
                    [str(t) for t in tags] if isinstance(tags, (list, tuple)) else str(tags),
                    ensure_ascii=False,
                )

        if 'lazyllm.io.input' in otel_attrs:
            attrs['langfuse.trace.input'] = otel_attrs['lazyllm.io.input']
        if 'lazyllm.io.output' in otel_attrs:
            attrs['langfuse.trace.output'] = otel_attrs['lazyllm.io.output']

    @staticmethod
    def _copy_trace_metadata(attrs: Dict[str, Any], otel_attrs: Dict[str, Any]) -> None:
        prefix = 'lazyllm.trace.metadata.'
        for key, value in otel_attrs.items():
            if key.startswith(prefix):
                attrs[f'langfuse.trace.metadata.{key[len(prefix):]}'] = value

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
        is_root_span = otel_attrs.get('lazyllm.span.is_root') is True

        semantic_type = otel_attrs.get('lazyllm.semantic_type')
        if semantic_type:
            langfuse_type = _SEMANTIC_TO_LANGFUSE_TYPE.get(semantic_type)
            if langfuse_type is None:
                if semantic_type not in self._warned_semantic_types:
                    LOG.warning(
                        f'LangfuseBackend: unmapped semantic_type {semantic_type!r}; '
                        f'falling back to "span".'
                    )
                    self._warned_semantic_types.add(semantic_type)
                langfuse_type = 'span'
            attrs['langfuse.observation.type'] = langfuse_type

        model = otel_attrs.get(
            'gen_ai.request.model',
            otel_attrs.get('lazyllm.entity.config.model'),
        )
        if semantic_type == SemanticType.LLM and model:
            attrs['gen_ai.request.model'] = str(model)

        if 'lazyllm.io.input' in otel_attrs:
            attrs['langfuse.observation.input'] = otel_attrs['lazyllm.io.input']
        if 'lazyllm.io.output' in otel_attrs:
            attrs['langfuse.observation.output'] = otel_attrs['lazyllm.io.output']

        if 'lazyllm.error.message' in otel_attrs:
            attrs['langfuse.observation.status_message'] = str(otel_attrs['lazyllm.error.message'])

        self._copy_usage_attrs(attrs, otel_attrs)

        if is_root_span:
            self._copy_trace_attrs(attrs, otel_attrs)
            self._copy_trace_metadata(attrs, otel_attrs)

        return attrs
