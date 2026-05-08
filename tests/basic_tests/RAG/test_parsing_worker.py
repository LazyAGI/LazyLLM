import threading
import time

import pytest

from lazyllm.tools.rag.parsing_service.base import TaskType
from lazyllm.tools.rag.parsing_service.worker import DocumentProcessorWorker


def _build_transfer_task(file_infos):
    return {
        'task_id': 'transfer-task',
        'task_type': TaskType.DOC_TRANSFER.value,
        'algo_id': '__default__',
        'kb_id': 'kb_source',
        'file_infos': file_infos,
    }


class _DummyProcessor:
    def __init__(self):
        self.calls = []

    def add_doc(self, **kwargs):
        self.calls.append(kwargs)


def test_explicit_transfer_payload_rejects_mixed_modes():
    impl = DocumentProcessorWorker._Impl()
    task = _build_transfer_task([
        {
            'file_path': '/tmp/a.txt',
            'doc_id': 'doc-a',
            'metadata': {},
            'transfer_params': {
                'mode': 'cp',
                'target_algo_id': '__default__',
                'target_doc_id': 'target-doc-a',
                'target_kb_id': 'kb_target',
            },
        },
        {
            'file_path': '/tmp/b.txt',
            'doc_id': 'doc-b',
            'metadata': {},
            'transfer_params': {
                'mode': 'mv',
                'target_algo_id': '__default__',
                'target_doc_id': 'target-doc-b',
                'target_kb_id': 'kb_target',
            },
        },
    ])

    with pytest.raises(ValueError, match='transfer_params.mode must be the same for all files'):
        impl._parse_task_payload(task)


def test_explicit_transfer_payload_rejects_mixed_target_kb():
    impl = DocumentProcessorWorker._Impl()
    task = _build_transfer_task([
        {
            'file_path': '/tmp/a.txt',
            'doc_id': 'doc-a',
            'metadata': {},
            'transfer_params': {
                'mode': 'cp',
                'target_algo_id': '__default__',
                'target_doc_id': 'target-doc-a',
                'target_kb_id': 'kb_target_a',
            },
        },
        {
            'file_path': '/tmp/b.txt',
            'doc_id': 'doc-b',
            'metadata': {},
            'transfer_params': {
                'mode': 'cp',
                'target_algo_id': '__default__',
                'target_doc_id': 'target-doc-b',
                'target_kb_id': 'kb_target_b',
            },
        },
    ])

    with pytest.raises(ValueError, match='transfer_params.target_kb_id must be the same for all files'):
        impl._parse_task_payload(task)


def test_explicit_transfer_payload_rejects_missing_transfer_params():
    impl = DocumentProcessorWorker._Impl()
    task = _build_transfer_task([
        {
            'file_path': '/tmp/a.txt',
            'doc_id': 'doc-a',
            'metadata': {},
            'transfer_params': None,
        },
    ])

    with pytest.raises(ValueError, match='transfer_params is required for task_type DOC_TRANSFER'):
        impl._parse_task_payload(task)


def test_explicit_transfer_payload_executes_with_single_mode_and_target():
    impl = DocumentProcessorWorker._Impl()
    task = _build_transfer_task([
        {
            'file_path': '/tmp/a.txt',
            'doc_id': 'doc-a',
            'metadata': {'k': 1},
            'transfer_params': {
                'mode': 'cp',
                'target_algo_id': '__default__',
                'target_doc_id': 'target-doc-a',
                'target_kb_id': 'kb_target',
            },
        },
        {
            'file_path': '/tmp/b.txt',
            'doc_id': 'doc-b',
            'metadata': {'k': 2},
            'transfer_params': {
                'mode': 'cp',
                'target_algo_id': '__default__',
                'target_doc_id': 'target-doc-b',
                'target_kb_id': 'kb_target',
            },
        },
    ])
    _, task_type, payload = impl._parse_task_payload(task)
    processor = _DummyProcessor()

    impl._exec_transfer_task(processor, 'transfer-task', payload)

    assert task_type == TaskType.DOC_TRANSFER.value
    assert processor.calls == [{
        'input_files': ['/tmp/a.txt', '/tmp/b.txt'],
        'ids': ['doc-a', 'doc-b'],
        'metadatas': [{'k': 1}, {'k': 2}],
        'kb_id': 'kb_source',
        'transfer_mode': 'cp',
        'target_kb_id': 'kb_target',
        'target_doc_ids': ['target-doc-a', 'target-doc-b'],
    }]


def test_worker_default_poll_mode_starts_thread_poller():
    poller_called = threading.Event()

    def _task_poller():
        poller_called.set()
        return []

    impl = DocumentProcessorWorker._Impl(task_poller=_task_poller)
    impl._lazy_init = lambda: None

    def _idle_worker():
        while not impl._shutdown:
            time.sleep(0.01)

    impl._worker_impl = _idle_worker

    try:
        impl.start()
        assert poller_called.wait(timeout=1.0) is True
        assert impl._poll_mode == 'thread'
        assert impl._poller_thread is not None
        assert impl._poller_thread.is_alive() is True
    finally:
        impl.shutdown()
