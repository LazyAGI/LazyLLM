'''Start standalone DocService.

Modes:
1. Full stack mode (default): starts algorithm registration, real
   DocumentProcessor, and DocServer in one process.
2. External parser mode: starts only DocServer and connects to an existing
   parsing service with ``--parser-url``.

Run:
    python examples/rag/doc_service_standalone.py --wait
    python examples/rag/doc_service_standalone.py --parser-url http://127.0.0.1:9966 --wait
'''

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

import requests

import lazyllm
from lazyllm import Document
from lazyllm.tools.rag.doc_service import DocServer
from lazyllm.tools.rag.doc_service.base import CallbackEventType, DocStatus, TaskCallbackRequest, TaskCreateRequest
from lazyllm.tools.rag.parsing_service import DocumentProcessor
from lazyllm.tools.rag.parsing_service.base import TaskStatus, TaskType
from lazyllm.tools.rag.utils import BaseResponse

REAL_ALGO_ID = 'real-standalone-algo'


def _make_db_config(db_name: str) -> Dict[str, Any]:
    return {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': db_name,
    }


def _wait_until(predicate, timeout: float = 20.0, interval: float = 0.1):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise RuntimeError(f'condition not satisfied before timeout, last={last!r}')


def _wait_http_ok(url: str, timeout: float = 20.0):
    def _poll():
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return resp
        except Exception:
            return None
        return None

    return _wait_until(_poll, timeout=timeout)


def _task_status_to_doc_status(task_status: str) -> DocStatus:
    mapping = {
        TaskStatus.SUCCESS.value: DocStatus.SUCCESS,
        TaskStatus.FAILED.value: DocStatus.FAILED,
        TaskStatus.CANCELED.value: DocStatus.CANCELED,
    }
    if task_status not in mapping:
        raise RuntimeError(f'unsupported task status: {task_status}')
    return mapping[task_status]


class _RealProcessorTaskAdapter:
    def __init__(self, parser_base_url: str, manager, upstream_client):
        self._parser_base_url = parser_base_url.rstrip('/')
        self._manager = manager
        self._upstream_client = upstream_client
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _load_doc(self, doc_id: str) -> Dict[str, Any]:
        doc = self._manager._get_doc(doc_id)
        if doc is None:
            raise RuntimeError(f'doc not found: {doc_id}')
        return doc

    @staticmethod
    def _load_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
        raw = doc.get('meta')
        return json.loads(raw) if raw else {}

    def _record_task(self, req: TaskCreateRequest) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        task = {
            'task_id': req.task_id,
            'task_type': req.task_type.value,
            'doc_id': req.doc_id,
            'kb_id': req.kb_id,
            'algo_id': req.algo_id,
            'status': TaskStatus.WAITING.value,
            'priority': req.priority,
            'callback_url': req.callback_url,
            'error_code': None,
            'error_msg': None,
            'created_at': now,
            'updated_at': now,
            'started_at': None,
            'finished_at': None,
        }
        with self._lock:
            self._tasks[req.task_id] = task
        return task

    def _dispatch_add_like_task(self, req: TaskCreateRequest, doc: Dict[str, Any], reparse: bool = False):
        file_info = {
            'file_path': doc['path'],
            'doc_id': req.doc_id,
            'metadata': self._load_metadata(doc),
        }
        if reparse:
            file_info['reparse_group'] = 'CoarseChunk'
        payload = {
            'task_id': req.task_id,
            'algo_id': req.algo_id,
            'kb_id': req.kb_id,
            'file_infos': [file_info],
            'priority': req.priority,
        }
        return requests.post(f'{self._parser_base_url}/doc/add', json=payload, timeout=15)

    def create_task(self, req: TaskCreateRequest):
        self._record_task(req)
        try:
            if req.task_type == TaskType.DOC_ADD:
                resp = self._dispatch_add_like_task(req, self._load_doc(req.doc_id))
            elif req.task_type == TaskType.DOC_REPARSE:
                resp = self._dispatch_add_like_task(req, self._load_doc(req.doc_id), reparse=True)
            elif req.task_type == TaskType.DOC_DELETE:
                resp = requests.delete(
                    f'{self._parser_base_url}/doc/delete',
                    json={
                        'task_id': req.task_id,
                        'algo_id': req.algo_id,
                        'kb_id': req.kb_id,
                        'doc_ids': [req.doc_id],
                        'priority': req.priority,
                    },
                    timeout=15,
                )
            elif req.task_type == TaskType.DOC_UPDATE_META:
                doc = self._load_doc(req.doc_id)
                resp = requests.post(
                    f'{self._parser_base_url}/doc/meta/update',
                    json={
                        'task_id': req.task_id,
                        'algo_id': req.algo_id,
                        'kb_id': req.kb_id,
                        'file_infos': [{
                            'file_path': doc['path'],
                            'doc_id': req.doc_id,
                            'metadata': req.metadata,
                        }],
                        'priority': req.priority,
                    },
                    timeout=15,
                )
            else:
                raise RuntimeError(f'unsupported task type: {req.task_type.value}')
            if resp.status_code >= 400:
                raise RuntimeError(f'parser http error: {resp.status_code} {resp.text}')
            result = BaseResponse.model_validate(resp.json())
            if result.code != 200:
                raise RuntimeError(f'parser task rejected: {result.msg}')
            return result
        except Exception:
            with self._lock:
                self._tasks.pop(req.task_id, None)
            raise

    def mark_task_finished(self, task_id: str, task_status: str,
                           error_code: Optional[str] = None, error_msg: Optional[str] = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            finished_at = datetime.now().isoformat()
            task['status'] = task_status
            task['error_code'] = error_code
            task['error_msg'] = error_msg
            task['finished_at'] = finished_at
            task['updated_at'] = finished_at
            return dict(task)

    def cancel_task(self, task_id: str):
        resp = requests.post(f'{self._parser_base_url}/doc/cancel', json={'task_id': task_id}, timeout=8)
        if resp.status_code >= 400:
            raise RuntimeError(f'parser http error: {resp.status_code} {resp.text}')
        result = BaseResponse.model_validate(resp.json())
        if result.code == 200 and result.data and result.data.get('cancel_status'):
            self.mark_task_finished(task_id, TaskStatus.CANCELED.value)
        return result

    def list_tasks(self, status: Optional[list[str]], page: int, page_size: int):
        with self._lock:
            items = [dict(task) for task in self._tasks.values()]
        if status:
            items = [item for item in items if item['status'] in status]
        items.sort(key=lambda item: item['created_at'], reverse=True)
        total = len(items)
        sliced = items[(page - 1) * page_size:page * page_size]
        return BaseResponse(
            code=200,
            msg='success',
            data={'items': sliced, 'total': total, 'page': page, 'page_size': page_size},
        )

    def get_task(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return BaseResponse(code=404, msg='task not found', data=None)
        return BaseResponse(code=200, msg='success', data=dict(task))

    def list_algorithms(self):
        return self._upstream_client.list_algorithms()

    def get_algorithm_groups(self, algo_id: str):
        return self._upstream_client.get_algorithm_groups(algo_id)


def _make_post_func(state: Dict[str, Any]):
    def _post_func(task_id: str, task_status: str, error_code: str = None, error_msg: str = None):
        adapter = state['adapter']
        callback_url = state['callback_url']
        task = adapter.mark_task_finished(task_id, task_status, error_code, error_msg)
        if task is None:
            raise RuntimeError(f'untracked callback task: {task_id}')
        callback = TaskCallbackRequest(
            callback_id=str(uuid4()),
            task_id=task_id,
            event_type=CallbackEventType.FINISH,
            status=_task_status_to_doc_status(task_status),
            error_code=error_code,
            error_msg=error_msg,
            payload={
                'task_type': task['task_type'],
                'doc_id': task['doc_id'],
                'kb_id': task['kb_id'],
                'algo_id': task['algo_id'],
            },
        )
        resp = requests.post(callback_url, json=callback.model_dump(mode='json'), timeout=8)
        resp.raise_for_status()
        return True

    return _post_func


def _build_store_conf(root_dir: str) -> Dict[str, Any]:
    segment_store_path = os.path.join(root_dir, 'segments.db')
    milvus_store_path = os.path.join(root_dir, 'milvus_lite.db')
    open(segment_store_path, 'a', encoding='utf-8').close()
    open(milvus_store_path, 'a', encoding='utf-8').close()
    return {
        'segment_store': {
            'type': 'map',
            'kwargs': {'uri': segment_store_path},
        },
        'vector_store': {
            'type': 'milvus',
            'kwargs': {
                'uri': milvus_store_path,
                'index_kwargs': {
                    'index_type': 'FLAT',
                    'metric_type': 'COSINE',
                },
            },
        },
    }


def _start_full_stack(args):
    tmp_dir = tempfile.mkdtemp(prefix='lazyllm_doc_service_standalone_')
    storage_dir = os.path.join(tmp_dir, 'uploads')
    os.makedirs(storage_dir, exist_ok=True)
    parser_db = os.path.join(tmp_dir, 'parser.db')
    doc_db = os.path.join(tmp_dir, 'doc_service.db')
    callback_state: Dict[str, Any] = {}

    parser = DocumentProcessor(
        port=args.parser_port,
        db_config=_make_db_config(parser_db),
        num_workers=args.num_workers,
        post_func=_make_post_func(callback_state),
    )
    parser.start()
    parser_base_url = parser._impl._url.rsplit('/', 1)[0]
    _wait_http_ok(f'{parser_base_url}/health')

    store_conf = _build_store_conf(tmp_dir)
    document = Document(
        dataset_path=None,
        name=args.algo_id,
        embed={'vec_dense': lambda x: [1.0, 2.0, 3.0]},
        store_conf=store_conf,
        display_name='Standalone Real Algo',
        manager=DocumentProcessor(url=parser_base_url),
        description='Algorithm registered by standalone doc service example',
    )
    document.create_node_group(
        name='line',
        transform=lambda x: x.split('\n'),
        parent='CoarseChunk',
        display_name='Line Chunk',
    )
    document.activate_group('CoarseChunk', embed_keys=['vec_dense'])
    document.activate_group('line', embed_keys=['vec_dense'])
    document.start()

    _wait_until(
        lambda: any(
            item.get('algo_id') == args.algo_id
            for item in requests.get(f'{parser_base_url}/algo/list', timeout=5).json().get('data', [])
        )
    )

    server = DocServer(
        storage_dir=storage_dir,
        db_config=_make_db_config(doc_db),
        parser_url=parser_base_url,
        port=args.port,
    )
    server.start()
    base_url = server.url.rsplit('/', 1)[0]
    _wait_http_ok(f'{base_url}/v1/health')

    raw_impl = server._raw_impl
    raw_impl._lazy_init()
    adapter = _RealProcessorTaskAdapter(
        parser_base_url=parser_base_url,
        manager=raw_impl._manager,
        upstream_client=raw_impl._manager._parser_client,
    )
    raw_impl._manager._parser_client = adapter
    callback_state['adapter'] = adapter
    callback_state['callback_url'] = raw_impl._manager._callback_url

    print(f'DocService URL: {base_url}', flush=True)
    print(f'DocService Docs: {base_url}/docs', flush=True)
    print(f'Parser URL: {parser_base_url}', flush=True)
    print(f'Parser Docs: {parser_base_url}/docs', flush=True)
    print(f'Algorithm ID: {args.algo_id}', flush=True)
    print(f'Storage Dir: {storage_dir}', flush=True)
    print(f'Doc DB: {doc_db}', flush=True)
    print(f'Parser DB: {parser_db}', flush=True)
    print(f'Tmp Dir: {tmp_dir}', flush=True)

    try:
        if args.wait:
            print('Full stack is running. Press Ctrl+C to stop...', flush=True)
            threading.Event().wait()
    finally:
        server.stop()
        try:
            parser.drop_algorithm(args.algo_id)
        except Exception:
            pass
        parser.stop()


def _start_doc_server_only(args):
    tmp_dir = tempfile.mkdtemp(prefix='lazyllm_doc_service_standalone_')
    storage_dir = os.path.join(tmp_dir, 'uploads')
    os.makedirs(storage_dir, exist_ok=True)
    doc_db = os.path.join(tmp_dir, 'doc_service.db')
    server = DocServer(
        storage_dir=storage_dir,
        db_config=_make_db_config(doc_db),
        parser_url=args.parser_url,
        port=args.port,
    )
    server.start()
    base_url = server.url.rsplit('/', 1)[0]
    print(f'DocService URL: {base_url}', flush=True)
    print(f'DocService Docs: {base_url}/docs', flush=True)
    print(f'Parser URL: {args.parser_url}', flush=True)
    print(f'Storage Dir: {storage_dir}', flush=True)
    print(f'Doc DB: {doc_db}', flush=True)

    try:
        if args.wait:
            print('DocService is running. Press Ctrl+C to stop...', flush=True)
            while True:
                time.sleep(1)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description='Standalone DocService server.')
    parser.add_argument('--port', type=int, default=8848, help='DocServer listen port.')
    parser.add_argument('--parser-port', type=int, default=9966, help='DocumentProcessor listen port.')
    parser.add_argument('--parser-url', type=str, default=None, help='Existing parsing service base URL.')
    parser.add_argument('--algo-id', type=str, default=REAL_ALGO_ID, help='Algorithm id to register in full stack mode.')
    parser.add_argument('--num-workers', type=int, default=1, help='DocumentProcessor worker count.')
    parser.add_argument('--wait', action='store_true', help='Keep server alive for manual API inspection.')
    args = parser.parse_args()

    if args.parser_url:
        _start_doc_server_only(args)
    else:
        _start_full_stack(args)


if __name__ == '__main__':
    main()
