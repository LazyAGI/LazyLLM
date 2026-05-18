import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

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


@pytest.mark.parametrize(
    'root_name, child_names, expected_count',
    [
        ('root-only', (), 1),
        ('root-one-child', ('child-a',), 2),
        ('root-many-children', tuple(f'step-{i}' for i in range(5)), 6),
    ],
)
def test_local_backend_writes_and_reads_trace(tmp_path, root_name, child_names, expected_count):
    from lazyllm.tracing.backends.local import LocalConsumeBackend

    trace_id = _emit_real_trace(tmp_path, root_name, child_names=child_names)

    backend = LocalConsumeBackend(storage_dir=tmp_path)
    lines = _read_trace_jsonl_lines(tmp_path, trace_id)
    payloads = [json.loads(line) for line in lines]
    raw = backend.fetch_trace_payload(trace_id)

    assert len(lines) == expected_count
    assert all('\n' not in line for line in lines)
    assert {payload['name'] for payload in payloads} == {root_name, *child_names}
    assert all(payload['context']['trace_id'] == f'0x{trace_id}' for payload in payloads)
    assert raw.trace.trace_id == trace_id and raw.trace.name == root_name
    assert {span.name for span in raw.spans} == {root_name, *child_names}
    assert all(span.trace_id == trace_id for span in raw.spans)
    assert all(not span.span_id.startswith('0x') for span in raw.spans)


def test_local_backend_consumes_concurrent_appends_to_same_trace(tmp_path):
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

    lines = _read_trace_jsonl_lines(tmp_path, TRACE_ID_A)
    payloads = [json.loads(line) for line in lines]
    raw = LocalConsumeBackend(storage_dir=tmp_path).fetch_trace_payload(TRACE_ID_A)

    assert len(lines) == num_spans and len(raw.spans) == num_spans
    assert {payload['name'] for payload in payloads} == {f'span-{idx}' for idx in range(1, num_spans + 1)}
    assert all(payload['context']['trace_id'] == f'0x{TRACE_ID_A}' for payload in payloads)
    assert {span.name for span in raw.spans} == {f'span-{idx}' for idx in range(1, num_spans + 1)}
    assert all(span.trace_id == TRACE_ID_A for span in raw.spans)
    assert all(span.parent_span_id == parent_span_id for span in raw.spans)


def test_consume_backend_rebuilds_raw_payload_from_local_jsonl(tmp_path):
    from lazyllm.tracing.backends.local import LocalConsumeBackend
    from lazyllm.tracing.consume.reconstruction import rebuild

    with _tracer_with_exporter(tmp_path) as tracer:
        with tracer.start_as_current_span('root') as root:
            root_context = root.get_span_context()
            trace_id = f'{root_context.trace_id:032x}'
            root_id = f'{root_context.span_id:016x}'
            root.set_attribute('lazyllm.span.kind', 'flow')
            root.set_attribute('lazyllm.span.is_root', True)
            root.set_attribute('lazyllm.status', 'ok')
            root.set_attribute('lazyllm.trace.name', 'root')
            root.set_attribute('lazyllm.trace.tags', ['local', 'poc'])
            root.set_attribute('lazyllm.trace.metadata.case', 'unit')
            root.set_attribute('session.id', 'session-1')
            root.set_attribute('user.id', 'user-1')

            with tracer.start_as_current_span('child') as child:
                child_id = f'{child.get_span_context().span_id:016x}'
                child.set_attribute('lazyllm.span.kind', 'callable')
                child.set_attribute('lazyllm.semantic_type', 'llm')
                child.set_attribute('lazyllm.status', 'ok')
                child.set_attribute('lazyllm.io.input', '{"args":["hello"],"kwargs":{}}')
                child.set_attribute('lazyllm.io.output', 'world')

    payload = LocalConsumeBackend(storage_dir=tmp_path).fetch_trace_payload(trace_id)
    structured = rebuild(payload.trace, payload.spans)
    stored_rows = [
        json.loads(line)
        for line in _read_trace_jsonl_lines(tmp_path, trace_id)
    ]
    stored_by_name = {row['name']: row for row in stored_rows}
    spans_by_id = {span.span_id: span for span in payload.spans}

    assert payload.trace.trace_id == trace_id
    assert payload.trace.name == 'root'
    assert payload.trace.session_id == 'session-1'
    assert payload.trace.user_id == 'user-1'
    assert payload.trace.tags == ['local', 'poc']
    assert payload.trace.metadata == {'case': 'unit'}
    assert stored_by_name['root']['context']['trace_id'] == f'0x{trace_id}'
    assert stored_by_name['root']['context']['span_id'] == f'0x{root_id}'
    assert stored_by_name['child']['context']['span_id'] == f'0x{child_id}'
    assert stored_by_name['child']['parent_id'] == f'0x{root_id}'
    assert {span.span_id for span in payload.spans} == {root_id, child_id}
    assert all(not span.span_id.startswith('0x') for span in payload.spans)
    assert spans_by_id[root_id].parent_span_id is None
    assert spans_by_id[child_id].parent_span_id == root_id
    assert spans_by_id[child_id].attributes['lazyllm.semantic_type'] == 'llm'
    assert spans_by_id[child_id].input == '{"args":["hello"],"kwargs":{}}'
    assert spans_by_id[child_id].output == 'world'
    assert structured.trace_id == trace_id
    assert structured.execution_tree.name == 'root'
    assert structured.execution_tree.children[0].name == 'child'
