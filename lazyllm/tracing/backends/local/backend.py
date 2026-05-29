import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from filelock import FileLock, Timeout

from lazyllm.common import LOG
from lazyllm.thirdparty import opentelemetry
from lazyllm.tracing.backends.base import ConsumeBackend, TracingBackend
from lazyllm.tracing.backends.utils import extract_trace_metadata, iso_to_epoch
from lazyllm.tracing.datamodel.raw import RawSpanRecord, RawTracePayload, RawTraceRecord
from lazyllm.tracing.errors import ConsumeBackendError, TraceNotFound
from lazyllm.tracing.semantics import is_valid_span_id, is_valid_trace_id

from .config import read_local_storage_dir


def _find_trace_path(storage_dir: Path, trace_id: str) -> Optional[Path]:
    paths = sorted(storage_dir.glob(f'*_{trace_id}.jsonl'))
    if len(paths) > 1:
        LOG.warning(
            f'Found multiple local trace files for trace_id={trace_id!r}: '
            f'{[path.name for path in paths]}'
        )
    return paths[0] if paths else None


def _trace_path(storage_dir: Path, trace_id: str, timestamp: Optional[str] = None) -> Path:
    path = _find_trace_path(storage_dir, trace_id)
    if path is not None:
        return path
    timestamp = timestamp or datetime.now().strftime('%Y%m%d%H%M%S')
    return storage_dir / f'{timestamp}_{trace_id}.jsonl'


def _trace_lock(storage_dir: Path, trace_id: str, timeout_seconds: Optional[float] = None) -> FileLock:
    timeout = timeout_seconds if timeout_seconds is not None else -1
    return FileLock(str(storage_dir / f'.{trace_id}.lock'), timeout=timeout)


def _strip_otel_id_prefix(value):
    return value[2:] if isinstance(value, str) and value.startswith('0x') else value


def _local_file_timestamp(otel_time: Any) -> Optional[str]:
    if not isinstance(otel_time, str) or not otel_time:
        return None
    try:
        text = otel_time[:-1] + '+00:00' if otel_time.endswith('Z') else otel_time
        return datetime.fromisoformat(text).astimezone().strftime('%Y%m%d%H%M%S')
    except ValueError:
        return None


def _span_to_json_line(span: 'opentelemetry.sdk.trace.ReadableSpan') -> Tuple[str, Optional[str]]:
    raw = span.to_json(indent=None)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f'span.to_json() must return JSON object, got {type(payload).__name__}')
    return json.dumps(payload, ensure_ascii=False), _local_file_timestamp(payload.get('start_time'))


def _choose_root_span(spans: List[RawSpanRecord]) -> Optional[RawSpanRecord]:
    for span in spans:
        if span.attributes.get('lazyllm.span.is_root') is True:
            return span
    for span in spans:
        if span.parent_span_id is None:
            return span
    return spans[0] if spans else None


def _raw_span_from_otel_json(row: Dict[str, Any]) -> RawSpanRecord:
    context = row.get('context')
    if not isinstance(context, dict):
        raise ConsumeBackendError('local span context field is missing or not a JSON object')

    trace_id = _strip_otel_id_prefix(context.get('trace_id'))
    span_id = _strip_otel_id_prefix(context.get('span_id'))
    if not is_valid_trace_id(trace_id) or not is_valid_span_id(span_id):
        raise ConsumeBackendError('local span context field is invalid')

    parent_span_id = _strip_otel_id_prefix(row.get('parent_id'))
    parent_span_id = parent_span_id if is_valid_span_id(parent_span_id) else None

    attributes = row.get('attributes')
    if not isinstance(attributes, dict):
        raise ConsumeBackendError('local span attributes field is not a JSON object')

    start_time = iso_to_epoch(row.get('start_time'))
    raw_end_time = row.get('end_time')
    end_time = iso_to_epoch(raw_end_time) if raw_end_time else None

    status_value = row.get('status')
    status_body = status_value if isinstance(status_value, dict) else {}
    status_code = status_body.get('status_code')
    lazy_status = attributes.get('lazyllm.status')
    status = 'error' if status_code == 'ERROR' or lazy_status == 'error' else 'ok'
    resource_value = row.get('resource')
    resource = resource_value if isinstance(resource_value, dict) else {}
    resource_attrs_value = resource.get('attributes')
    resource_attrs = resource_attrs_value if isinstance(resource_attrs_value, dict) else {}

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
        end_time=end_time,
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


def _read_trace_file(
    storage_dir: Path,
    trace_id: str,
    timeout_seconds: Optional[float],
) -> tuple[str, List[str]]:
    try:
        with _trace_lock(storage_dir, trace_id, timeout_seconds=timeout_seconds):
            path = _find_trace_path(storage_dir, trace_id)
            if path is None:
                raise TraceNotFound(trace_id)
            with path.open('r', encoding='utf-8') as file_obj:
                return path.name, file_obj.readlines()
    except Timeout as exc:
        raise ConsumeBackendError(f'timed out waiting for local trace file lock: {trace_id}') from exc


def _raw_spans_from_lines(path_name: str, lines: List[str]) -> List[RawSpanRecord]:
    spans = []
    for line_no, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except ValueError as exc:
            raise ConsumeBackendError(
                f'invalid JSON in local trace file {path_name} at line {line_no}'
            ) from exc
        if not isinstance(row, dict):
            raise ConsumeBackendError(
                f'local trace file {path_name} line {line_no} is not a JSON object'
            )

        try:
            spans.append(_raw_span_from_otel_json(row))
        except (ConsumeBackendError, ValueError) as exc:
            raise ConsumeBackendError(
                f'invalid span in local trace file {path_name} at line {line_no}: {exc}'
            ) from exc
    return spans


class LocalFileSpanExporter(opentelemetry.sdk.trace.export.SpanExporter):
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else read_local_storage_dir()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown = False

    def export(self, spans: Sequence['opentelemetry.sdk.trace.ReadableSpan']):
        SpanExportResult = opentelemetry.sdk.trace.export.SpanExportResult
        if self._shutdown:
            return SpanExportResult.FAILURE

        grouped: Dict[str, List[str]] = {}
        timestamps: Dict[str, str] = {}
        try:
            for span in spans:
                context = span.get_span_context()
                trace_id = f'{context.trace_id:032x}'
                line, timestamp = _span_to_json_line(span)
                grouped.setdefault(trace_id, []).append(line)
                if timestamp is not None:
                    timestamps.setdefault(trace_id, timestamp)
        except Exception as exc:
            LOG.warning(f'LocalFileSpanExporter failed to serialize spans: {exc}')
            return SpanExportResult.FAILURE

        try:
            for trace_id, lines in grouped.items():
                with _trace_lock(self.storage_dir, trace_id):
                    path = _trace_path(self.storage_dir, trace_id, timestamps.get(trace_id))
                    with path.open('a', encoding='utf-8') as file_obj:
                        file_obj.writelines(f'{line}\n' for line in lines)
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
        # Local exports keep the original OTel attributes in the JSONL payload.
        return {}


class LocalConsumeBackend(ConsumeBackend):
    name = 'local'

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else read_local_storage_dir()

    def fetch_trace_payload(self, trace_id: str, *, timeout_seconds: Optional[float] = None) -> RawTracePayload:
        if not is_valid_trace_id(trace_id):
            raise ValueError(f'invalid trace_id: {trace_id!r}')

        path_name, lines = _read_trace_file(self.storage_dir, trace_id, timeout_seconds)
        spans = _raw_spans_from_lines(path_name, lines)
        return RawTracePayload(
            trace=_raw_trace_from_spans(trace_id, spans),
            spans=spans,
        )
