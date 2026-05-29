import json
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from filelock import FileLock, Timeout

from lazyllm.common import LOG
from lazyllm.thirdparty import opentelemetry
from lazyllm.tracing.backends.base import ConsumeBackend, TracingBackend
from lazyllm.tracing.backends.utils import extract_trace_metadata, iso_to_epoch
from lazyllm.tracing.datamodel.raw import RawSpanRecord, RawTracePayload, RawTraceRecord
from lazyllm.tracing.errors import ConsumeBackendError, TraceNotFound
from lazyllm.tracing.semantics import is_valid_span_id, is_valid_trace_id

from .config import (
    read_local_archive_retention_seconds,
    read_local_archive_seconds,
    read_local_storage_dir,
)


_TRACE_TIME_FORMAT = '%Y%m%d%H%M%S'
_MAINTENANCE_LOCK_NAME = '.maintenance.lock'
_ARCHIVE_INDEX_NAME = '.archive_index.json'
_ARCHIVE_INDEX_LOCK_NAME = '.archive_index.lock'
_DEFAULT_MAINTENANCE_TIMEOUT_SECONDS = 2


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
    index = _read_archive_index(storage_dir)
    indexed_entry = _archive_index_entry(index, trace_id)
    if indexed_entry is not None:
        archive_name, member_name = indexed_entry
        restored_path = storage_dir / f'{datetime.now().strftime(_TRACE_TIME_FORMAT)}_{trace_id}.jsonl'
        tmp_path = restored_path.with_suffix('.jsonl.tmp')
        try:
            with zipfile.ZipFile(storage_dir / archive_name) as zip_file:
                with zip_file.open(member_name) as source, tmp_path.open('wb') as target:
                    target.write(source.read())
            tmp_path.replace(restored_path)
            _update_archive_index(storage_dir, lambda current: current.pop(trace_id, None))
            return restored_path
        except (FileNotFoundError, KeyError, OSError, zipfile.BadZipFile) as exc:
            tmp_path.unlink(missing_ok=True)
            LOG.warning(
                f'Failed to restore indexed local trace archive '
                f'{archive_name}:{member_name}: {exc}'
            )
    timestamp = timestamp or datetime.now().strftime(_TRACE_TIME_FORMAT)
    return storage_dir / f'{timestamp}_{trace_id}.jsonl'


def _maintenance_lock(storage_dir: Path, timeout_seconds: Optional[float] = None) -> FileLock:
    timeout = timeout_seconds if timeout_seconds is not None else _DEFAULT_MAINTENANCE_TIMEOUT_SECONDS
    return FileLock(str(storage_dir / _MAINTENANCE_LOCK_NAME), timeout=timeout)


def _archive_index_lock(storage_dir: Path) -> FileLock:
    return FileLock(str(storage_dir / _ARCHIVE_INDEX_LOCK_NAME), timeout=-1)


@contextmanager
def _local_trace_lock(storage_dir: Path, trace_id: str, timeout_seconds: Optional[float] = None):
    timeout = -1 if timeout_seconds is None or timeout_seconds < 0 else timeout_seconds
    trace_lock = FileLock(str(storage_dir / f'.{trace_id}.lock'), timeout=timeout)
    try:
        with trace_lock:
            yield
    except Timeout as exc:
        raise ConsumeBackendError(f'timed out waiting for local trace file lock: {trace_id}') from exc


def _parse_prefix_time(path: Path) -> Optional[datetime]:
    prefix = path.name.split('_', 1)[0] if path.suffix == '.jsonl' else path.stem
    try:
        return datetime.strptime(prefix, _TRACE_TIME_FORMAT)
    except ValueError:
        LOG.warning(f'Invalid local trace timestamp prefix: {path.name}')
        return None


def _is_older_than(path: Path, now: datetime, seconds: int) -> bool:
    timestamp = _parse_prefix_time(path)
    return timestamp is not None and timestamp <= now - timedelta(seconds=seconds)


def _next_archive_path(storage_dir: Path, now: datetime) -> Path:
    timestamp = now
    while True:
        # Keep archive names timestamp-sortable; on same-second collisions, try the next second.
        path = storage_dir / f'{timestamp.strftime(_TRACE_TIME_FORMAT)}.zip'
        if not path.exists() and not path.with_suffix('.zip.tmp').exists():
            return path
        timestamp += timedelta(seconds=1)


def _warn_duplicate_trace_ids(paths: List[Path]) -> None:
    trace_files: Dict[str, List[str]] = {}
    for path in paths:
        if path.name.endswith('.jsonl') and '_' in path.name:
            trace_id = path.name.rsplit('_', 1)[1][:-len('.jsonl')]
            trace_files.setdefault(trace_id, []).append(path.name)
    for trace_id, names in trace_files.items():
        if len(names) > 1:
            LOG.warning(f'Found duplicate local trace files for trace_id={trace_id!r}: {names}')


def _read_archive_index(storage_dir: Path) -> Dict[str, Dict[str, str]]:
    path = storage_dir / _ARCHIVE_INDEX_NAME
    try:
        with path.open('r', encoding='utf-8') as file_obj:
            data = json.load(file_obj)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        LOG.warning(f'Failed to read local trace archive index {path.name}: {exc}')
        return {}

    return data if isinstance(data, dict) else {}


def _write_archive_index(storage_dir: Path, index: Dict[str, Dict[str, str]]) -> None:
    path = storage_dir / _ARCHIVE_INDEX_NAME
    tmp_path = path.with_name(f'{path.name}.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as file_obj:
            json.dump(index, file_obj, ensure_ascii=False, sort_keys=True)
            file_obj.write('\n')
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        LOG.warning(f'Failed to write local trace archive index {path.name}')


def _update_archive_index(storage_dir: Path, update: Callable[[Dict[str, Dict[str, str]]], None]) -> None:
    with _archive_index_lock(storage_dir):
        index = _read_archive_index(storage_dir)
        update(index)
        _write_archive_index(storage_dir, index)


def _archive_index_entry(index: Dict[str, Any], trace_id: str) -> Optional[Tuple[str, str]]:
    entry = index.get(trace_id)
    if not isinstance(entry, dict):
        return None
    archive_name = entry.get('archive')
    member_name = entry.get('member')
    if (
        isinstance(archive_name, str) and isinstance(member_name, str) and
        Path(archive_name).name == archive_name and archive_name.endswith('.zip') and
        Path(member_name).name == member_name and member_name.endswith(f'_{trace_id}.jsonl')
    ):
        return archive_name, member_name
    return None


def _archive_old_jsonl(
    storage_dir: Path,
    now: datetime,
    archive_seconds: int,
    timeout_seconds: Optional[float] = None,
) -> List[str]:
    jsonl_paths = sorted(
        path for path in storage_dir.glob('*.jsonl')
        if _is_older_than(path, now, archive_seconds)
    )
    if not jsonl_paths:
        return []

    _warn_duplicate_trace_ids(jsonl_paths)
    archive_path = _next_archive_path(storage_dir, now)
    tmp_path = archive_path.with_suffix('.zip.tmp')
    archived_paths: Dict[Path, Tuple[int, int]] = {}
    try:
        with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for path in jsonl_paths:
                if '_' not in path.name:
                    LOG.warning(f'Invalid local trace file name: {path.name}')
                    continue
                _, trace_id = path.stem.split('_', 1)
                try:
                    with _local_trace_lock(storage_dir, trace_id, timeout_seconds):
                        stat_result = path.stat()
                        zip_file.write(path, arcname=path.name)
                    archived_paths[path] = (stat_result.st_mtime_ns, stat_result.st_size)
                except ConsumeBackendError as exc:
                    LOG.warning(f'Skipping archiving for trace {trace_id} due to lock timeout: {exc}')

        if not archived_paths:
            tmp_path.unlink(missing_ok=True)
            return []

        tmp_path.replace(archive_path)  # Publish the completed archive atomically.
        for path in archived_paths:
            _, trace_id = path.stem.split('_', 1)
            try:
                with _local_trace_lock(storage_dir, trace_id, timeout_seconds):
                    stat_result = path.stat()
                    if archived_paths[path] != (stat_result.st_mtime_ns, stat_result.st_size):
                        continue
                    path.unlink(missing_ok=True)
                    _update_archive_index(
                        storage_dir,
                        lambda index: index.update({
                            trace_id: {'archive': archive_path.name, 'member': path.name},
                        }),
                    )
            except OSError as exc:
                LOG.warning(f'Failed to delete archived trace file {path.name}: {exc}')
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return [path.name for path in archived_paths]


def _delete_expired_archives(storage_dir: Path, now: datetime, retention_seconds: int) -> List[str]:
    deleted = []
    for path in sorted(storage_dir.glob('*.zip')):
        if not _is_older_than(path, now, retention_seconds):
            continue
        try:
            path.unlink()
            deleted.append(path.name)
        except OSError as exc:
            LOG.warning(f'Failed to delete expired archive {path.name}: {exc}')
    return deleted


def maintain_local_traces(
    storage_dir: Optional[Path] = None,
    timeout_seconds: Optional[float] = _DEFAULT_MAINTENANCE_TIMEOUT_SECONDS,
) -> Dict[str, List[str]]:
    target_dir = Path(storage_dir) if storage_dir is not None else read_local_storage_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    result = {'compressed_jsonl': [], 'deleted_zip': []}

    try:
        with _maintenance_lock(target_dir, timeout_seconds):
            now = datetime.now()
            result['compressed_jsonl'] = _archive_old_jsonl(
                target_dir,
                now,
                read_local_archive_seconds(),
                timeout_seconds=timeout_seconds,
            )
            result['deleted_zip'] = _delete_expired_archives(
                target_dir,
                now,
                read_local_archive_retention_seconds(),
            )
            if result['deleted_zip'] and (target_dir / _ARCHIVE_INDEX_NAME).exists():
                deleted_zip = set(result['deleted_zip'])

                def remove_deleted_zip(index):
                    for trace_id, entry in list(index.items()):
                        if not isinstance(entry, dict) or entry.get('archive') in deleted_zip:
                            index.pop(trace_id, None)

                _update_archive_index(target_dir, remove_deleted_zip)
    except Timeout as exc:
        raise ConsumeBackendError(f'timed out waiting for local trace maintenance lock: {target_dir}') from exc

    return result


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
    with _local_trace_lock(storage_dir, trace_id, timeout_seconds=timeout_seconds):
        path = _find_trace_path(storage_dir, trace_id)
        if path is None:
            return _read_archived_trace_file(storage_dir, trace_id)
        try:
            with path.open('r', encoding='utf-8') as file_obj:
                return path.name, file_obj.readlines()
        except FileNotFoundError:
            return _read_archived_trace_file(storage_dir, trace_id)


def _read_archived_trace_file(storage_dir: Path, trace_id: str) -> tuple[str, List[str]]:
    indexed_entry = _archive_index_entry(_read_archive_index(storage_dir), trace_id)
    if indexed_entry is not None:
        archive_name, member_name = indexed_entry
        try:
            with zipfile.ZipFile(storage_dir / archive_name) as zip_file:
                with zip_file.open(member_name) as file_obj:
                    lines = [line.decode('utf-8') for line in file_obj.readlines()]
            return f'{archive_name}:{member_name}', lines
        except (FileNotFoundError, KeyError, OSError, zipfile.BadZipFile) as exc:
            LOG.warning(
                f'Failed to read indexed local trace archive '
                f'{archive_name}:{member_name}: {exc}'
            )

    archive_paths = [
        (timestamp, path)
        for path in storage_dir.glob('*.zip')
        if (timestamp := _parse_prefix_time(path)) is not None
    ]
    for _, archive_path in sorted(archive_paths, key=lambda item: item[0], reverse=True):
        try:
            with zipfile.ZipFile(archive_path) as zip_file:
                matched_names = sorted(
                    (
                        name for name in zip_file.namelist()
                        if name.endswith(f'_{trace_id}.jsonl')
                    ),
                    reverse=True,
                )
                if not matched_names:
                    continue
                if len(matched_names) > 1:
                    LOG.warning(
                        f'Found multiple archived local trace files for trace_id={trace_id!r} '
                        f'in {archive_path.name}: {matched_names}'
                    )
                name = matched_names[0]
                with zip_file.open(name) as file_obj:
                    lines = [line.decode('utf-8') for line in file_obj.readlines()]
                _update_archive_index(
                    storage_dir,
                    lambda index: index.update({
                        trace_id: {'archive': archive_path.name, 'member': name},
                    }),
                )
                return f'{archive_path.name}:{name}', lines
        except (FileNotFoundError, zipfile.BadZipFile) as exc:
            LOG.warning(f'Failed to read local trace archive {archive_path.name}: {exc}')

    LOG.warning(f'trace_id does not exist or has been cleaned: {trace_id}')
    raise TraceNotFound(trace_id)


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
                with _local_trace_lock(self.storage_dir, trace_id):
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
