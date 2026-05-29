import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta

from lazyllm import config
from lazyllm.thirdparty import opentelemetry


TRACE_ID_A = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'


def _read_trace_jsonl_lines(storage_dir, trace_id):
    stored_files = list(storage_dir.glob(f'*_{trace_id}.jsonl'))
    assert len(stored_files) == 1
    return stored_files[0].read_text(encoding='utf-8').splitlines()


@contextmanager
def _tracer_with_exporter(storage_dir):
    from lazyllm.tracing.backends.local.backend import LocalFileSpanExporter

    exporter = LocalFileSpanExporter(storage_dir=storage_dir)
    provider = opentelemetry.sdk.trace.TracerProvider()
    provider.add_span_processor(opentelemetry.sdk.trace.export.SimpleSpanProcessor(exporter))
    try:
        yield provider.get_tracer('lazyllm.tracing.local.test')
    finally:
        try:
            provider.force_flush()
        finally:
            provider.shutdown()


def _emit_real_trace(storage_dir, root_name, child_names=()):
    with _tracer_with_exporter(storage_dir) as tracer:
        with tracer.start_as_current_span(root_name) as root:
            root.set_attribute('lazyllm.span.kind', 'flow')
            root.set_attribute('lazyllm.span.is_root', True)
            root.set_attribute('lazyllm.status', 'ok')
            trace_id = f'{root.get_span_context().trace_id:032x}'
            for child_name in child_names:
                with tracer.start_as_current_span(child_name) as child:
                    child.set_attribute('lazyllm.span.kind', 'callable')
                    child.set_attribute('lazyllm.status', 'ok')
    return trace_id


def test_local_backend_writes_and_reads_trace(tmp_path):
    from lazyllm.tracing.backends.local import LocalConsumeBackend

    trace_id = _emit_real_trace(tmp_path, 'root', child_names=('child-a', 'child-b'))

    backend = LocalConsumeBackend(storage_dir=tmp_path)
    lines = _read_trace_jsonl_lines(tmp_path, trace_id)
    payloads = [json.loads(line) for line in lines]
    raw = backend.fetch_trace_payload(trace_id)

    assert len(lines) == 3
    assert {p['name'] for p in payloads} == {'root', 'child-a', 'child-b'}
    assert all(p['context']['trace_id'] == f'0x{trace_id}' for p in payloads)

    assert raw.trace.trace_id == trace_id
    assert raw.trace.name == 'root'
    assert {s.name for s in raw.spans} == {'root', 'child-a', 'child-b'}
    assert all(s.trace_id == trace_id for s in raw.spans)


def test_local_backend_concurrent_appends_to_same_trace(tmp_path):
    from lazyllm.tracing.backends.local import LocalConsumeBackend

    with _tracer_with_exporter(tmp_path) as tracer:
        parent_span_id = '1111111111111111'
        parent_context = opentelemetry.trace.set_span_in_context(
            opentelemetry.trace.NonRecordingSpan(
                opentelemetry.trace.SpanContext(
                    trace_id=int(TRACE_ID_A, 16),
                    span_id=int(parent_span_id, 16),
                    is_remote=False,
                    trace_flags=opentelemetry.trace.TraceFlags(0x01),
                )
            )
        )

        def emit_one(idx):
            with tracer.start_as_current_span(f'span-{idx}', context=parent_context) as span:
                span.set_attribute('lazyllm.span.kind', 'callable')
                span.set_attribute('lazyllm.status', 'ok')

        num_spans = 64
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(emit_one, range(1, num_spans + 1)))

    payload = LocalConsumeBackend(storage_dir=tmp_path).fetch_trace_payload(TRACE_ID_A)
    assert len(payload.spans) == num_spans
    assert {s.name for s in payload.spans} == {f'span-{i}' for i in range(1, num_spans + 1)}
    assert all(s.parent_span_id == parent_span_id for s in payload.spans)


def test_local_backend_archive_lifecycle(tmp_path):
    from lazyllm.tracing.backends.local import LocalConsumeBackend, maintain_local_traces

    trace_id = _emit_real_trace(tmp_path, 'archived-root', child_names=('archived-child',))
    fresh_trace_id = _emit_real_trace(tmp_path, 'fresh-root')

    original_file = next(tmp_path.glob(f'*_{trace_id}.jsonl'))
    old_prefix = (datetime.now() - timedelta(seconds=20)).strftime('%Y%m%d%H%M%S')
    archived_source = tmp_path / f'{old_prefix}_{trace_id}.jsonl'
    original_file.rename(archived_source)
    fresh_source = next(tmp_path.glob(f'*_{fresh_trace_id}.jsonl'))

    expired_zip = tmp_path / f'{(datetime.now() - timedelta(seconds=60)).strftime("%Y%m%d%H%M%S")}.zip'
    with zipfile.ZipFile(expired_zip, 'w') as zf:
        zf.writestr('expired.jsonl', '')

    with config.temp('trace_local_archive_seconds', 10), \
            config.temp('trace_local_archive_retention_seconds', 30):
        result = maintain_local_traces(tmp_path)
        fresh_payload = LocalConsumeBackend(storage_dir=tmp_path).fetch_trace_payload(fresh_trace_id)

    assert result['compressed_jsonl'] == [archived_source.name]
    assert result['deleted_zip'] == [expired_zip.name]
    assert not archived_source.exists()
    assert fresh_source.exists()
    assert fresh_payload.trace.trace_id == fresh_trace_id

    archive_name = next(p.name for p in tmp_path.glob('*.zip'))
    index = json.loads((tmp_path / '.archive_index.json').read_text(encoding='utf-8'))
    assert index[trace_id] == {'archive': archive_name, 'member': archived_source.name}

    # archived trace can still be read
    payload = LocalConsumeBackend(storage_dir=tmp_path).fetch_trace_payload(trace_id)
    assert payload.trace.trace_id == trace_id
    assert {s.name for s in payload.spans} == {'archived-root', 'archived-child'}

    # expired archive is cleaned up
    with config.temp('trace_local_archive_seconds', 10), \
            config.temp('trace_local_archive_retention_seconds', 0):
        result = maintain_local_traces(tmp_path)
    assert result['deleted_zip'] == [archive_name]
    assert trace_id not in json.loads((tmp_path / '.archive_index.json').read_text(encoding='utf-8'))


def test_local_backend_keeps_live_jsonl_changed_during_archive(tmp_path, monkeypatch):
    from lazyllm.tracing.backends.local import maintain_local_traces

    trace_id = _emit_real_trace(tmp_path, 'changing-root')
    original_file = next(tmp_path.glob(f'*_{trace_id}.jsonl'))
    old_prefix = (datetime.now() - timedelta(seconds=20)).strftime('%Y%m%d%H%M%S')
    archived_source = tmp_path / f'{old_prefix}_{trace_id}.jsonl'
    original_file.rename(archived_source)
    extra_line = archived_source.read_text(encoding='utf-8').splitlines()[0]

    original_write = zipfile.ZipFile.write

    def write_and_change_file(zip_file, filename, *args, **kwargs):
        result = original_write(zip_file, filename, *args, **kwargs)
        if filename == archived_source:
            with archived_source.open('a', encoding='utf-8') as file_obj:
                file_obj.write(f'{extra_line}\n')
        return result

    monkeypatch.setattr(zipfile.ZipFile, 'write', write_and_change_file)

    with config.temp('trace_local_archive_seconds', 10):
        maintain_local_traces(tmp_path)

    assert archived_source.exists()
    assert len(archived_source.read_text(encoding='utf-8').splitlines()) == 2
    index_path = tmp_path / '.archive_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    assert trace_id not in index


def test_local_backend_restores_archived_trace_before_late_append(tmp_path):
    from lazyllm.tracing.backends.local import LocalConsumeBackend, maintain_local_traces

    trace_id = _emit_real_trace(tmp_path, 'late-root')
    original_file = next(tmp_path.glob(f'*_{trace_id}.jsonl'))
    old_prefix = (datetime.now() - timedelta(seconds=20)).strftime('%Y%m%d%H%M%S')
    archived_source = tmp_path / f'{old_prefix}_{trace_id}.jsonl'
    original_file.rename(archived_source)

    with config.temp('trace_local_archive_seconds', 10):
        maintain_local_traces(tmp_path)

    index_path = tmp_path / '.archive_index.json'
    assert not list(tmp_path.glob(f'*_{trace_id}.jsonl'))
    assert trace_id in json.loads(index_path.read_text(encoding='utf-8'))

    with _tracer_with_exporter(tmp_path) as tracer:
        parent_context = opentelemetry.trace.set_span_in_context(
            opentelemetry.trace.NonRecordingSpan(
                opentelemetry.trace.SpanContext(
                    trace_id=int(trace_id, 16),
                    span_id=int('2222222222222222', 16),
                    is_remote=False,
                    trace_flags=opentelemetry.trace.TraceFlags(0x01),
                )
            )
        )
        with tracer.start_as_current_span('late-child', context=parent_context) as span:
            span.set_attribute('lazyllm.span.kind', 'callable')
            span.set_attribute('lazyllm.status', 'ok')

    restored_files = list(tmp_path.glob(f'*_{trace_id}.jsonl'))
    payload = LocalConsumeBackend(storage_dir=tmp_path).fetch_trace_payload(trace_id)

    assert len(restored_files) == 1
    assert restored_files[0].name != archived_source.name
    assert trace_id not in json.loads(index_path.read_text(encoding='utf-8'))
    assert {s.name for s in payload.spans} == {'late-root', 'late-child'}
