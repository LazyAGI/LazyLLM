import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from filelock import FileLock, Timeout

from lazyllm.common import LOG
from lazyllm.thirdparty import opentelemetry
from lazyllm.tracing.backends.base import ConsumeBackend, TracingBackend
from lazyllm.tracing.backends.utils import extract_trace_metadata, iso_to_epoch
from lazyllm.tracing.datamodel.raw import RawSpanRecord, RawTracePayload, RawTraceRecord
from lazyllm.tracing.errors import ConsumeBackendError, TraceNotFound
from lazyllm.tracing.semantics import is_valid_span_id, is_valid_trace_id

from .config import read_local_storage_dir


def _trace_path(storage_dir: Path, trace_id: str) -> Path:
    return storage_dir / f'{trace_id}.jsonl'


def _trace_lock(path: Path, timeout_seconds: Optional[float] = None) -> FileLock:
    timeout = timeout_seconds if timeout_seconds is not None else -1
    return FileLock(str(path) + '.lock', timeout=timeout)


def _strip_otel_id_prefix(value: str) -> str:
    return value[2:] if value.startswith('0x') else value


def _span_to_json_line(span: 'opentelemetry.sdk.trace.ReadableSpan') -> str:
    raw = span.to_json(indent=None)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f'span.to_json() must return JSON object, got {type(payload).__name__}')
    return json.dumps(payload, ensure_ascii=False)


def _choose_root_span(spans: List[RawSpanRecord]) -> Optional[RawSpanRecord]:
    for span in spans:
        if span.attributes.get('lazyllm.span.is_root') is True:
            return span
    for span in spans:
        if span.parent_span_id is None:
            return span
    return spans[0] if spans else None


def _raw_span_from_otel_json(row: Dict[str, Any]) -> RawSpanRecord:
    trace_id = _strip_otel_id_prefix(row['context'].get('trace_id'))
    span_id = _strip_otel_id_prefix(row['context'].get('span_id'))
    if not is_valid_trace_id(trace_id) or not is_valid_span_id(span_id):
        raise ConsumeBackendError('local span context field is invalid')
    parent_span_id = None
    parent = row.get('parent_id')
    if parent not in (None, ''):
        candidate = _strip_otel_id_prefix(parent)
        if is_valid_span_id(candidate):
            parent_span_id = candidate

    attributes = row.get('attributes')
    if attributes is None:
        attributes = {}
    if not isinstance(attributes, dict):
        raise ConsumeBackendError('local span attributes field is not a JSON object')

    start_time = iso_to_epoch(row.get('start_time'))
    if start_time is None:
        raise ConsumeBackendError('local span missing start_time')

    status_body = row.get('status') if isinstance(row.get('status'), dict) else {}
    status_code = status_body.get('status_code')
    lazy_status = attributes.get('lazyllm.status')
    status = 'error' if status_code == 'ERROR' or lazy_status == 'error' else 'ok'
    resource = row.get('resource') if isinstance(row.get('resource'), dict) else {}
    resource_attrs = resource.get('attributes') if isinstance(resource.get('attributes'), dict) else {}

    metadata = {
        'attributes': dict(attributes),
        'resourceAttributes': dict(resource_attrs),
    }

    return RawSpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=str(row.get('name') or ''),
        start_time=start_time,
        end_time=iso_to_epoch(row.get('end_time')),
        status=status,
        attributes=dict(attributes),
        input=attributes.get('lazyllm.io.input'),
        output=attributes.get('lazyllm.io.output'),
        metadata=metadata,
        error_message=attributes.get('lazyllm.error.message') or status_body.get('description'),
        raw=dict(row),
    )


def _raw_trace_from_spans(trace_id: str, spans: List[RawSpanRecord]) -> RawTraceRecord:
    root = _choose_root_span(spans)
    root_attrs = root.attributes if root is not None else {}
    start_time = min((span.start_time for span in spans), default=None)
    end_values = [span.end_time for span in spans if span.end_time is not None]
    end_time = max(end_values) if end_values else None
    status = 'error' if any(span.status == 'error' for span in spans) else 'ok'
    tags = root_attrs.get('lazyllm.trace.tags')
    if not isinstance(tags, list):
        tags = []
    metadata = extract_trace_metadata(root_attrs)

    return RawTraceRecord(
        trace_id=trace_id,
        name=root_attrs.get('lazyllm.trace.name') or (root.name if root is not None else trace_id),
        session_id=root_attrs.get('session.id'),
        user_id=root_attrs.get('user.id'),
        tags=[str(item) for item in tags],
        metadata=metadata,
        input=root_attrs.get('lazyllm.io.input'),
        output=root_attrs.get('lazyllm.io.output'),
        start_time=start_time,
        end_time=end_time,
        status=status,
        raw={'backend': 'local', 'span_count': len(spans)},
    )


class LocalFileSpanExporter(opentelemetry.sdk.trace.export.SpanExporter):
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else read_local_storage_dir()
        self._shutdown = False

    def export(self, spans: Sequence['opentelemetry.sdk.trace.ReadableSpan']):
        SpanExportResult = opentelemetry.sdk.trace.export.SpanExportResult
        if self._shutdown:
            return SpanExportResult.FAILURE

        grouped: Dict[str, List[str]] = {}
        try:
            for span in spans:
                context = span.get_span_context()
                trace_id = f'{context.trace_id:032x}'
                grouped.setdefault(trace_id, []).append(_span_to_json_line(span))
        except Exception as exc:
            LOG.warning(f'LocalFileSpanExporter failed to serialize spans: {exc}')
            return SpanExportResult.FAILURE

        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            for trace_id, lines in grouped.items():
                path = _trace_path(self.storage_dir, trace_id)
                with _trace_lock(path):
                    with path.open('a', encoding='utf-8') as file_obj:
                        file_obj.write(''.join(line + '\n' for line in lines))
        except Exception as exc:
            LOG.warning(f'LocalFileSpanExporter failed to write spans: {exc}')
            return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class LocalBackend(TracingBackend):
    name = 'local'

    def build_exporter(self):
        return LocalFileSpanExporter()

    def map_attributes(self, otel_attrs: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class LocalConsumeBackend(ConsumeBackend):
    name = 'local'

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else read_local_storage_dir()

    def fetch_trace_payload(self, trace_id: str, *, timeout_seconds: Optional[float] = None) -> RawTracePayload:
        if not is_valid_trace_id(trace_id):
            raise ValueError(f'invalid trace_id: {trace_id!r}')

        path = _trace_path(self.storage_dir, trace_id)
        if not path.exists():
            raise TraceNotFound(trace_id)

        rows: List[Dict[str, Any]] = []
        try:
            with _trace_lock(path, timeout_seconds=timeout_seconds):
                with path.open('r', encoding='utf-8') as file_obj:
                    for line_no, line in enumerate(file_obj, start=1):
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            row = json.loads(text)
                        except ValueError as exc:
                            raise ConsumeBackendError(
                                f'invalid JSON in local trace file {path.name} at line {line_no}'
                            ) from exc
                        if not isinstance(row, dict):
                            raise ConsumeBackendError(
                                f'local trace file {path.name} line {line_no} is not a JSON object'
                            )
                        rows.append(row)
        except Timeout as exc:
            raise ConsumeBackendError(f'timed out waiting for local trace file lock: {path.name}') from exc

        spans = [_raw_span_from_otel_json(row) for row in rows]
        spans = [span for span in spans if span.trace_id == trace_id]
        spans.sort(key=lambda span: (span.start_time, span.name, span.span_id))
        return RawTracePayload(
            trace=_raw_trace_from_spans(trace_id, spans),
            spans=spans,
        )
