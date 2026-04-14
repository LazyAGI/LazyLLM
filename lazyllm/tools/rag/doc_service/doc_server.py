from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Set

import requests

from lazyllm import LOG, FastapiApp as app, ModuleBase, ServerModule, UrlModule, once_wrapper
from lazyllm.thirdparty import fastapi

from ..utils import BaseResponse, _get_default_db_config, ensure_call_endpoint
from .base import (
    AddFileItem,
    AddRequest,
    AlgorithmInfoRequest,
    CallbackEventType,
    DeleteRequest,
    DocServiceError,
    DocStatus,
    KbBatchQueryRequest,
    KbCreateRequest,
    KbDeleteBatchRequest,
    KbUpdateRequest,
    MetadataPatchRequest,
    ReparseRequest,
    SourceType,
    TaskBatchRequest,
    TaskCallbackPayload,
    TaskCallbackRequest,
    TaskCancelRequest,
    TaskInfoRequest,
    TransferRequest,
    UploadRequest,
)
from .doc_manager import DocManager
from .utils import sha256_file

DEFAULT_OPENAPI_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'doc_server.openapi.json')


class DocServer(ModuleBase):
    class _Impl:
        def __init__(
            self,
            storage_dir: str,
            db_config: Optional[Dict[str, Any]] = None,
            parser_db_config: Optional[Dict[str, Any]] = None,
            parser_poll_interval: float = 0.05,
            parser_url: Optional[str] = None,
            callback_url: Optional[str] = None,
            enable_scan: bool = False,
            scan_interval: int = 10,
        ):
            if not parser_url:
                raise ValueError('parser_url is required; doc_service no longer starts a mock parsing server')
            self._storage_dir = storage_dir
            self._db_config = db_config
            self._parser_db_config = parser_db_config
            self._parser_poll_interval = parser_poll_interval
            self._parser_url = parser_url
            self._callback_url = callback_url
            self._parser = None
            self._manager = None
            self._enable_scan = enable_scan
            self._scan_interval = scan_interval
            self._scan_thread = None
            self._scan_continue = False
            self._owned_kbs: Set[str] = set()

        @once_wrapper(reset_on_pickle=True)
        def _lazy_init(self):
            if self._storage_dir and not os.path.exists(self._storage_dir):
                os.makedirs(self._storage_dir, exist_ok=True)
            self._manager = DocManager(
                db_config=self._db_config,
                parser_url=self._parser_url,
                callback_url=self._callback_url,
            )
            # NOTE: scanning is NOT started here.  Use ``enable_scanning()`` after
            # all KB registrations and parser algorithm registrations are complete
            # so the first scan sees a consistent _owned_kbs set and can route
            # requests to algorithms that actually exist on the parser.
            # For standalone / direct DocServer usage with enable_scan=True and no
            # explicit ``enable_scanning()`` call, the scan thread is started lazily
            # on the first ``_sync_dataset()`` invocation if still not running.

        def stop(self):
            self._scan_continue = False
            if self._scan_thread and self._scan_thread.is_alive():
                self._scan_thread.join(timeout=2)
            return None

        def _sync_dataset_for_kb(self, kb_id: str, algo_id: str, disk_files: list, disk_set: set):
            '''Sync one KB: diff disk vs documents table -> upload new / delete stale via unified pipeline.'''
            # For retry: exclude FAILED/CANCELED so they get re-uploaded
            synced_docs = self._manager._list_kb_docs_by_path(kb_id, exclude_failed=True)
            # For stale cleanup: include FAILED/CANCELED so removed files get cleaned up
            all_known_docs = self._manager._list_kb_docs_by_path(kb_id, exclude_failed=False)

            # New files (or previously failed) → upload(source_type=SCAN)
            new_paths = [p for p in disk_files if p not in synced_docs]
            if new_paths:
                try:
                    request = UploadRequest(
                        items=[AddFileItem(file_path=p) for p in new_paths],
                        kb_id=kb_id, algo_id=algo_id, source_type=SourceType.SCAN,
                    )
                    self._manager.upload(request)
                except Exception as exc:
                    LOG.error(f'[Scan] upload failed for kb={kb_id}: {len(new_paths)} files: {exc}')

            # Stale files (including failed ones whose source file was removed) → delete
            stale_ids = [did for path, did in all_known_docs.items() if path not in disk_set]
            if stale_ids:
                try:
                    request = DeleteRequest(doc_ids=stale_ids, kb_id=kb_id, algo_id=algo_id)
                    self._manager.delete(request)
                except Exception as exc:
                    LOG.error(f'[Scan] delete failed for kb={kb_id}: {len(stale_ids)} docs: {exc}')

            if new_paths or stale_ids:
                LOG.info(f'[Scan] kb={kb_id} sync done: added={len(new_paths)}, deleted={len(stale_ids)}')

        def _sync_dataset(self):
            '''One-shot scan: list dir -> sync all active KB+algo pairs.'''
            from .utils import list_dataset_files
            disk_files = list_dataset_files(self._storage_dir)
            disk_set = set(disk_files)

            kb_algo_pairs = self._manager._list_active_kb_algo_pairs()
            if not kb_algo_pairs:
                kb_algo_pairs = [('__default__', '__default__')]

            # When this instance has explicitly registered KBs, only scan those
            # to avoid processing KBs that belong to other Document instances
            # sharing the same global DB.
            owned = self._owned_kbs.copy()
            if owned:
                kb_algo_pairs = [(kb, algo) for kb, algo in kb_algo_pairs if kb in owned]

            for kb_id, algo_id in kb_algo_pairs:
                try:
                    self._sync_dataset_for_kb(kb_id, algo_id, disk_files, disk_set)
                except Exception as exc:
                    LOG.error(f'[Scan] sync failed for kb={kb_id}, algo={algo_id}: {exc}')

        def _scan_worker(self):
            '''Daemon thread: periodically scan dataset directory.'''
            while self._scan_continue:
                try:
                    self._sync_dataset()
                except Exception as exc:
                    LOG.error(f'[Scan] sync failed: {exc}')
                time.sleep(self._scan_interval)

        def _start_scan_monitoring(self):
            if self._scan_thread and self._scan_thread.is_alive():
                return
            self._scan_continue = True
            self._scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
            self._scan_thread.start()

        def enable_scanning(self):
            '''Start scanning after all KB registrations and parser algo registrations
            are complete.  Safe to call multiple times (idempotent).

            This is the intended way for ``Document._Manager`` to trigger the first
            scan: it ensures ``_owned_kbs`` is fully populated and all algorithms
            have been registered with the parser before any file-level sync happens.
            '''
            self._lazy_init()
            if not self._enable_scan:
                return
            if not (self._storage_dir and os.path.isdir(self._storage_dir)):
                return
            self._sync_dataset()
            self._start_scan_monitoring()

        def ensure_kb_registered(self, kb_id: str, algo_id: Optional[str] = None):
            '''Lightweight KB registration: ensure KB + algo binding rows exist in DB.

            Unlike ``create_kb_by_id`` this does NOT validate algorithm existence
            against the parser, so it can be called before the algorithm is registered
            (e.g. during ``add_kb_group`` which creates a DocImpl that will register
            its algorithm later during ``_lazy_init``).
            '''
            self._lazy_init()
            algo_id = algo_id or kb_id
            self._manager._ensure_kb(kb_id, display_name=kb_id)
            self._manager._ensure_kb_algorithm(kb_id, algo_id)
            self._owned_kbs.add(kb_id)

        def set_runtime_callback_url(self, callback_url: str):
            self._lazy_init()
            self._manager.set_callback_url(callback_url)

        @staticmethod
        def _response(data=None, code=200, msg='success', status_code=200):
            payload = BaseResponse(code=code, msg=msg, data=data).model_dump(mode='json')
            return fastapi.responses.JSONResponse(status_code=status_code, content=payload)

        def _run(self, func, *args, success_msg='success', **kwargs):
            try:
                data = func(*args, **kwargs)
                return self._response(data=data, msg=success_msg)
            except DocServiceError as exc:
                data = dict(exc.data or {})
                data.setdefault('biz_code', exc.biz_code)
                return self._response(data=data, code=exc.http_status, msg=exc.msg, status_code=exc.http_status)
            except fastapi.HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                data = detail.get('data')
                if isinstance(data, dict) and 'biz_code' not in data and detail.get('code'):
                    data['biz_code'] = detail['code']
                code = exc.status_code
                msg = detail.get('msg', str(exc.detail))
                return self._response(data=data, code=code, msg=msg, status_code=exc.status_code)

        @staticmethod
        def _build_upload_payload(request: UploadRequest, file_identities: Optional[List[Dict[str, Any]]] = None):
            source_type = request.source_type or SourceType.API
            items = file_identities
            if items is None:
                items = []
                for idx, item in enumerate(request.items):
                    content_hash = None
                    size_bytes = None
                    if os.path.exists(item.file_path):
                        content_hash = sha256_file(item.file_path)
                        size_bytes = os.path.getsize(item.file_path)
                    items.append({
                        'filename': os.path.basename(item.file_path),
                        'content_hash': content_hash,
                        'size_bytes': size_bytes,
                        'doc_id': item.doc_id if idx == 0 else None,
                    })
            return {
                'kb_id': request.kb_id,
                'algo_id': request.algo_id,
                'source_type': source_type.value,
                'idempotency_key': request.idempotency_key,
                'items': items,
            }

        @staticmethod
        def _build_update_kb_payload(kb_id: str, request: KbUpdateRequest):
            payload = request.model_dump(mode='json', exclude_unset=True)
            payload['kb_id'] = kb_id
            payload['explicit_fields'] = sorted(field for field in request.model_fields_set if field != 'kb_id')
            return payload

        def _gen_unique_upload_path(self, filename: str, reserved_paths: Optional[set] = None):
            safe_name = os.path.basename(filename) or 'upload.bin'
            file_path = os.path.join(self._storage_dir, safe_name)
            reserved_paths = reserved_paths or set()
            if file_path not in reserved_paths and not os.path.exists(file_path):
                return file_path

            suffix = os.path.splitext(safe_name)[1]
            prefix = safe_name[:-len(suffix)] if suffix else safe_name
            for idx in range(1, 10000):
                candidate = os.path.join(self._storage_dir, f'{prefix}-{idx}{suffix}')
                if candidate not in reserved_paths and not os.path.exists(candidate):
                    return candidate
            digest = hashlib.sha256(safe_name.encode()).hexdigest()[:8]
            return os.path.join(self._storage_dir, f'{prefix}-{digest}{suffix}')

        @staticmethod
        async def _save_upload_file(upload_file: 'fastapi.UploadFile', file_path: str):
            with open(file_path, 'wb') as fh:
                while True:
                    chunk = await upload_file.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            await upload_file.close()

        async def _persist_uploads(self, files: List['fastapi.UploadFile']):
            saved_paths = []
            file_identities = []
            reserved_paths: Set[str] = set()
            for upload_file in files:
                filename = getattr(upload_file, 'filename', None) or 'upload.bin'
                file_path = self._gen_unique_upload_path(filename, reserved_paths)
                await self._save_upload_file(upload_file, file_path)
                reserved_paths.add(file_path)
                saved_paths.append(file_path)
                file_identities.append({
                    'filename': os.path.basename(file_path),
                    'content_hash': sha256_file(file_path),
                    'size_bytes': os.path.getsize(file_path),
                    'doc_id': None,
                })
            return saved_paths, file_identities

        def _run_upload(self, request: UploadRequest, payload: Optional[Dict[str, Any]] = None):
            idem_payload = payload or self._build_upload_payload(request)
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/upload', request.idempotency_key, idem_payload,
                lambda: {'items': self._manager.upload(request)}
            ))

        @staticmethod
        def _normalize_task_callback(callback: Any) -> TaskCallbackRequest:
            if isinstance(callback, TaskCallbackRequest):
                return callback
            if not isinstance(callback, dict):
                raise DocServiceError('E_INVALID_PARAM', 'invalid callback payload')

            payload = dict(callback.get('payload') or {})
            for field in ('task_type', 'doc_id', 'kb_id', 'algo_id'):
                if callback.get(field) is not None and field not in payload:
                    payload[field] = callback[field]

            event_type = callback.get('event_type')
            status = callback.get('status')
            task_status = callback.get('task_status')

            try:
                if status is not None:
                    normalized_status = DocStatus(status)
                    normalized_event_type = CallbackEventType(event_type) if event_type else (
                        CallbackEventType.START
                        if normalized_status in (DocStatus.WAITING, DocStatus.WORKING) else CallbackEventType.FINISH
                    )
                elif task_status is not None:
                    normalized_status = DocStatus(task_status)
                    normalized_event_type = CallbackEventType(event_type) if event_type else (
                        CallbackEventType.START
                        if normalized_status in (DocStatus.WAITING, DocStatus.WORKING)
                        else CallbackEventType.FINISH
                    )
                else:
                    raise DocServiceError('E_INVALID_PARAM', 'status or task_status is required')
            except ValueError as exc:
                raise DocServiceError('E_INVALID_PARAM', str(exc)) from exc

            callback_data = {
                'callback_id': callback.get('callback_id'),
                'task_id': callback.get('task_id'),
                'event_type': normalized_event_type,
                'status': normalized_status,
                'error_code': callback.get('error_code'),
                'error_msg': callback.get('error_msg'),
                'payload': payload,
            }
            return TaskCallbackRequest.model_validate({k: v for k, v in callback_data.items() if v is not None})

        @staticmethod
        def _format_task_view(task: Optional[Dict[str, Any]]):
            if not isinstance(task, dict):
                return task
            return dict(task)

        def _format_task_response_data(self, data: Any):
            if isinstance(data, dict) and isinstance(data.get('items'), list):
                payload = dict(data)
                payload['items'] = [self._format_task_view(item) for item in data['items']]
                return payload
            return self._format_task_view(data)

        def upload_request(self, request: UploadRequest):
            self._lazy_init()
            return self._run_upload(request)

        @app.post('/v1/docs/upload')
        async def upload(
            self,
            files: List['fastapi.UploadFile'] = fastapi.File(...),  # noqa: B008
            kb_id: Optional[str] = fastapi.Form(None),  # noqa: B008
            algo_id: Optional[str] = fastapi.Form(None),  # noqa: B008
            source_type: Optional[SourceType] = fastapi.Form(None),  # noqa: B008
            doc_id: Optional[str] = fastapi.Form(None),  # noqa: B008
            idempotency_key: Optional[str] = fastapi.Form(None),  # noqa: B008
        ):
            self._lazy_init()
            if not files:
                raise fastapi.HTTPException(status_code=400, detail='files is required')
            kb_id = kb_id or '__default__'
            algo_id = algo_id or '__default__'
            source_type = source_type or SourceType.API
            saved_paths, file_identities = await self._persist_uploads(files)
            upload_request = UploadRequest(
                items=[
                    AddFileItem(file_path=path, doc_id=(doc_id if idx == 0 else None))
                    for idx, path in enumerate(saved_paths)
                ],
                kb_id=kb_id,
                algo_id=algo_id,
                source_type=source_type,
                idempotency_key=idempotency_key,
            )
            if file_identities:
                file_identities[0]['doc_id'] = doc_id
            return self._run_upload(upload_request, self._build_upload_payload(upload_request, file_identities))

        @app.post('/v1/docs/add')
        def add(self, request: AddRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/add', request.idempotency_key, payload, lambda: {'items': self._manager.add_files(request)}
            ))

        @app.post('/v1/docs/reparse')
        def reparse(self, request: ReparseRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/reparse', request.idempotency_key, payload,
                lambda: {'task_ids': self._manager.reparse(request)}
            ))

        @app.post('/v1/docs/delete')
        def delete(self, request: DeleteRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/delete', request.idempotency_key, payload, lambda: {'items': self._manager.delete(request)}
            ))

        @app.post('/v1/docs/transfer')
        def transfer(self, request: TransferRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/transfer', request.idempotency_key, payload,
                lambda: {'items': self._manager.transfer(request)}
            ))

        @app.get('/v1/docs')
        def list_docs(
            self,
            status: Optional[List[str]] = None,
            kb_id: Optional[str] = None,
            algo_id: Optional[str] = None,
            keyword: Optional[str] = None,
            include_deleted_or_canceled: bool = True,
            page: int = 1,
            page_size: int = 20,
        ):
            self._lazy_init()
            data = self._manager.list_docs(
                status=status,
                kb_id=kb_id,
                algo_id=algo_id,
                keyword=keyword,
                include_deleted_or_canceled=include_deleted_or_canceled,
                page=page,
                page_size=page_size,
            )
            return BaseResponse(code=200, msg='success', data=data)

        @app.get('/v1/docs/{doc_id}')
        def get_doc(self, doc_id: str):
            self._lazy_init()
            return self._run(lambda: self._manager.get_doc_detail(doc_id))

        @app.post('/v1/docs/metadata/patch')
        def patch_metadata(self, request: MetadataPatchRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/metadata/patch', request.idempotency_key, payload,
                lambda: self._manager.patch_metadata(request)
            ))

        @app.get('/v1/tasks')
        def list_tasks(self, status: Optional[List[str]] = None, page: int = 1, page_size: int = 20):
            self._lazy_init()
            resp = self._manager.list_tasks(status, page, page_size)
            return self._response(
                data=self._format_task_response_data(resp.data),
                code=resp.code,
                msg=resp.msg,
                status_code=resp.code,
            )

        @app.get('/v1/tasks/{task_id}')
        def get_task(self, task_id: str):
            self._lazy_init()
            resp = self._manager.get_task(task_id)
            return self._response(
                data=self._format_task_response_data(resp.data),
                code=resp.code,
                msg=resp.msg,
                status_code=resp.code,
            )

        def cancel_task_by_id(self, task_id: str):
            self._lazy_init()
            resp = self._manager.cancel_task(task_id)
            return self._response(data=resp.data, code=resp.code, msg=resp.msg, status_code=resp.code)

        @app.post('/v1/tasks/cancel')
        def cancel_task(self, request: TaskCancelRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')

            def _cancel():
                resp = self._manager.cancel_task(request.task_id)
                if resp.code == 404:
                    raise DocServiceError('E_NOT_FOUND', resp.msg, resp.data)
                if resp.code == 409:
                    raise DocServiceError('E_STATE_CONFLICT', resp.msg, resp.data)
                if resp.code != 200:
                    raise DocServiceError('E_INVALID_PARAM', resp.msg, resp.data)
                return resp.data
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/tasks/cancel', request.idempotency_key, payload, _cancel
            ))

        def task_callback(self, callback: Any):
            self._lazy_init()
            return self._run(lambda: self._manager.on_task_callback(self._normalize_task_callback(callback)))

        @app.post('/v1/internal/callbacks/tasks')
        def task_callback_http(self, request: TaskCallbackPayload):
            return self.task_callback(request.model_dump(mode='json', exclude_none=True))

        @app.get('/v1/algo/list')
        def list_algo(self):
            self._lazy_init()
            return self._run(lambda: self._manager.list_algorithms())

        @app.get('/v1/algo/{algo_id}/groups')
        def get_algo_groups(self, algo_id: str):
            self._lazy_init()
            return self._run(lambda: self._manager.get_algo_groups(algo_id))

        @app.get('/v1/algorithms')
        def list_algorithms(self):
            self._lazy_init()
            return self._run(lambda: self._manager.list_algorithms_compat())

        def list_algorithms_impl(self):
            self._lazy_init()
            return self._run(lambda: self._manager.list_algorithms_compat())

        @app.post('/v1/algorithms/info')
        def get_algorithm_info(self, request: AlgorithmInfoRequest):
            self._lazy_init()
            return self._run(lambda: self._manager.get_algorithm_info(request.algo_id))

        def get_algorithm_info_impl(self, algo_id: str):
            self._lazy_init()
            return self._run(lambda: self._manager.get_algorithm_info(algo_id))

        @app.get('/v1/chunks')
        def list_chunks(
            self,
            kb_id: str,
            doc_id: str,
            group: str,
            algo_id: str = '__default__',
            page: int = 1,
            page_size: int = 20,
            offset: Optional[int] = None,
        ):
            self._lazy_init()
            return self._run(lambda: self._manager.list_chunks(
                kb_id=kb_id, doc_id=doc_id, group=group, algo_id=algo_id,
                page=page, page_size=page_size, offset=offset,
            ))

        @app.post('/v1/tasks/batch')
        def get_tasks_batch(self, request: TaskBatchRequest):
            self._lazy_init()
            return self._run(lambda: self._manager.get_tasks_batch(request.task_ids))

        def get_tasks_batch_impl(self, task_ids: List[str]):
            self._lazy_init()
            return self._run(lambda: self._manager.get_tasks_batch(task_ids))

        @app.post('/v1/tasks/info')
        def get_task_info(self, request: TaskInfoRequest):
            self._lazy_init()
            resp = self._manager.get_task(request.task_id)
            return self._response(
                data=self._format_task_response_data(resp.data),
                code=resp.code,
                msg=resp.msg,
                status_code=resp.code,
            )

        def get_task_info_impl(self, task_id: str):
            self._lazy_init()
            resp = self._manager.get_task(task_id)
            return self._response(
                data=self._format_task_response_data(resp.data),
                code=resp.code,
                msg=resp.msg,
                status_code=resp.code,
            )

        @app.get('/v1/kbs')
        def list_kbs(
            self,
            page: int = 1,
            page_size: int = 20,
            keyword: Optional[str] = None,
            status: Optional[List[str]] = None,
            owner_id: Optional[str] = None,
        ):
            self._lazy_init()
            return self._run(lambda: self._manager.list_kbs(
                page=page,
                page_size=page_size,
                keyword=keyword,
                status=status,
                owner_id=owner_id,
            ))

        @app.get('/v1/kbs/{kb_id}')
        def get_kb(self, kb_id: str):
            self._lazy_init()
            return self._run(lambda: self._manager.get_kb(kb_id))

        def create_kb_by_id(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                            owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                            algo_id: str = '__default__'):
            self._lazy_init()
            return self._run(lambda: self._manager.create_kb(
                kb_id,
                display_name=display_name,
                description=description,
                owner_id=owner_id,
                meta=meta,
                algo_id=algo_id,
            ))

        @app.post('/v1/kbs')
        def create_kb(self, request: KbCreateRequest):
            self._lazy_init()
            if not request.kb_id:
                raise DocServiceError('E_INVALID_PARAM', 'kb_id is required')
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/kbs', request.idempotency_key, payload,
                lambda: self._manager.create_kb(
                    request.kb_id,
                    display_name=request.display_name,
                    description=request.description,
                    owner_id=request.owner_id,
                    meta=request.meta,
                    algo_id=request.algo_id,
                )
            ))

        def update_kb_by_id(self, kb_id: str, request: KbUpdateRequest):
            self._lazy_init()
            if request.kb_id and request.kb_id != kb_id:
                raise DocServiceError(
                    'E_INVALID_PARAM',
                    f'kb_id mismatch: path={kb_id}, body={request.kb_id}',
                    {'kb_id': kb_id, 'request_kb_id': request.kb_id},
                )
            payload = self._build_update_kb_payload(kb_id, request)
            return self._run(lambda: self._manager.run_idempotent(
                f'/v1/kbs/{kb_id}:patch', request.idempotency_key, payload,
                lambda: self._manager.update_kb(
                    kb_id,
                    display_name=request.display_name,
                    description=request.description,
                    owner_id=request.owner_id,
                    meta=request.meta,
                    algo_id=request.algo_id,
                    explicit_fields=set(request.model_fields_set),
                )
            ))

        @app.post('/v1/kbs/{kb_id}/update')
        def update_kb(self, kb_id: str, request: KbUpdateRequest):
            return self.update_kb_by_id(kb_id, request)

        @app.post('/v1/kbs/batch')
        def batch_get_kbs(self, request: KbBatchQueryRequest):
            self._lazy_init()
            return self._run(lambda: self._manager.batch_get_kbs(request.kb_ids))

        @app.delete('/v1/kbs/{kb_id}')
        def delete_kb(self, kb_id: str, idempotency_key: Optional[str] = None):
            self._lazy_init()
            payload = {'kb_id': kb_id}
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/kbs/{kb_id}:delete', idempotency_key, payload, lambda: self._manager.delete_kb(kb_id)
            ))

        @app.delete('/v1/kbs')
        def delete_kbs(self, request: KbDeleteBatchRequest):
            self._lazy_init()
            payload = request.model_dump(mode='json')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/kbs:delete', request.idempotency_key, payload, lambda: self._manager.delete_kbs(request.kb_ids)
            ))

        def delete_kbs_impl(self, kb_ids: List[str], idempotency_key: Optional[str] = None):
            self._lazy_init()
            payload = {'kb_ids': kb_ids}
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/kbs:delete', idempotency_key, payload, lambda: self._manager.delete_kbs(kb_ids)
            ))

        @app.get('/v1/health')
        def health(self):
            self._lazy_init()
            return BaseResponse(code=200, msg='success', data=self._manager.health())

        @app.get('/v1/internal/parser-url')
        def get_parser_url(self):
            self._lazy_init()
            return BaseResponse(code=200, msg='success', data={'parser_url': self._parser_url})

        def __call__(self, func_name: str, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

    def __init__(
        self,
        port: Optional[int] = None,
        url: Optional[str] = None,
        parser_url: Optional[str] = None,
        db_config: Optional[Dict[str, Any]] = None,
        parser_db_config: Optional[Dict[str, Any]] = None,
        parser_poll_interval: float = 0.05,
        storage_dir: Optional[str] = None,
        callback_url: Optional[str] = None,
        pythonpath: Optional[str] = None,
        launcher=None,
        enable_scan: bool = False,
        scan_interval: int = 10,
    ):
        super().__init__()
        self._raw_impl = None
        self._storage_dir = storage_dir or os.path.join(os.getcwd(), '.doc_service_uploads')
        self._db_config = db_config or _get_default_db_config('doc_service')
        self._parser_db_config = parser_db_config or _get_default_db_config('doc_service_parser')
        if url:
            self._impl = UrlModule(url=ensure_call_endpoint(url))
        else:
            if not parser_url:
                raise ValueError('parser_url is required; doc_service no longer embeds a mock parsing server')
            self._raw_impl = DocServer._Impl(
                storage_dir=self._storage_dir,
                db_config=self._db_config,
                parser_db_config=self._parser_db_config,
                parser_poll_interval=parser_poll_interval,
                parser_url=parser_url,
                callback_url=callback_url,
                enable_scan=enable_scan,
                scan_interval=scan_interval,
            )
            self._impl = ServerModule(self._raw_impl, port=port, launcher=launcher, pythonpath=pythonpath)

    @staticmethod
    def _register_openapi_routes(openapi_app: 'fastapi.FastAPI', impl: 'DocServer._Impl'):
        def _find_services(cls):
            if '__relay_services__' not in dir(cls):
                return
            if '__relay_services__' in cls.__dict__:
                for (method, path), (name, kw) in cls.__relay_services__.items():
                    if getattr(impl.__class__, name) is getattr(cls, name):
                        route_method = getattr(openapi_app, 'get' if method == 'list' else method)
                        route_method(path, **kw)(getattr(impl, name))
            for base in cls.__bases__:
                _find_services(base)

        app.update()
        _find_services(impl.__class__)

    @classmethod
    def build_openapi_app(cls, title: str = 'LazyLLM DocService API', version: str = '1.0.0'):
        openapi_app = fastapi.FastAPI(
            title=title,
            version=version,
            description='OpenAPI schema generated from current DocServer routes.',
        )
        impl = cls._Impl(
            storage_dir=os.path.join(os.getcwd(), '.doc_service_openapi'),
            parser_url='http://127.0.0.1:9966',
        )
        cls._register_openapi_routes(openapi_app, impl)
        for route in openapi_app.routes:
            body_field = getattr(route, 'body_field', None)
            annotation = getattr(getattr(body_field, 'field_info', None), 'annotation', None)
            if hasattr(annotation, 'model_rebuild'):
                annotation.model_rebuild(force=True, _types_namespace=route.endpoint.__globals__)
        return openapi_app

    @classmethod
    def build_openapi_schema(cls, title: str = 'LazyLLM DocService API', version: str = '1.0.0'):
        return cls.build_openapi_app(title=title, version=version).openapi()

    @classmethod
    def export_openapi(
        cls,
        output_path: str = DEFAULT_OPENAPI_OUTPUT_PATH,
        title: str = 'LazyLLM DocService API',
        version: str = '1.0.0',
    ):
        schema = cls.build_openapi_schema(title=title, version=version)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as fh:
            json.dump(schema, fh, ensure_ascii=False, indent=2, sort_keys=True)
        return output_path

    def start(self):
        result = super().start()
        if self._raw_impl and isinstance(self._impl, ServerModule):
            try:
                callback_url = self._impl._url.rsplit('/', 1)[0] + '/v1/internal/callbacks/tasks'
                self._dispatch('set_runtime_callback_url', callback_url)
            except Exception as exc:
                LOG.warning(f'[DocServer] failed to set runtime callback url: {exc}')
        return result

    def stop(self):
        if self._raw_impl:
            try:
                self._dispatch('stop')
            except Exception as exc:
                LOG.warning(f'[DocServer] stop impl failed: {exc}, {traceback.format_exc()}')
        if isinstance(self._impl, ServerModule):
            self._impl.stop()

    @property
    def url(self):
        return self._impl._url

    @property
    def _url(self):
        return self.url

    @property
    def parser_url(self):
        if self._raw_impl:
            return self._raw_impl._parser_url
        base_url = self.url.rsplit('/', 1)[0]
        try:
            response = requests.get(f'{base_url}/v1/internal/parser-url', timeout=5)
            response.raise_for_status()
            return response.json()['data']['parser_url']
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            LOG.warning(f'[DocServer] failed to resolve remote parser_url from {base_url}: {exc}')
            return None

    @staticmethod
    def _normalize_dispatch_result(result):
        if isinstance(result, fastapi.responses.JSONResponse):
            return json.loads(result.body.decode())
        return result

    def _dispatch(self, method: str, *args, **kwargs):
        if isinstance(self._impl, ServerModule):
            return self._normalize_dispatch_result(self._impl._call(method, *args, **kwargs))
        return self._normalize_dispatch_result(getattr(self._impl, method)(*args, **kwargs))

    # Method-call style wrappers
    def upload(self, request: UploadRequest):
        return self._dispatch('upload_request', request)

    def add(self, request: AddRequest):
        return self._dispatch('add', request)

    def reparse(self, request: ReparseRequest):
        return self._dispatch('reparse', request)

    def delete(self, request: DeleteRequest):
        return self._dispatch('delete', request)

    def transfer(self, request: TransferRequest):
        return self._dispatch('transfer', request)

    def patch_metadata(self, request: MetadataPatchRequest):
        return self._dispatch('patch_metadata', request)

    def list_docs(self, **kwargs):
        return self._dispatch('list_docs', **kwargs)

    def get_doc(self, doc_id: str):
        return self._dispatch('get_doc', doc_id)

    def list_tasks(self, **kwargs):
        return self._dispatch('list_tasks', **kwargs)

    def get_tasks_batch(self, task_ids: List[str]):
        return self._dispatch('get_tasks_batch_impl', task_ids)

    def get_task_info(self, task_id: str):
        return self._dispatch('get_task_info_impl', task_id)

    def get_task(self, task_id: str):
        return self._dispatch('get_task', task_id)

    def set_runtime_callback_url(self, callback_url: str):
        return self._dispatch('set_runtime_callback_url', callback_url)

    def cancel_task(self, task_id: str):
        return self._dispatch('cancel_task_by_id', task_id)

    def list_kbs(self, **kwargs):
        return self._dispatch('list_kbs', **kwargs)

    def get_kb(self, kb_id: str):
        return self._dispatch('get_kb', kb_id)

    def list_chunks(self, **kwargs):
        return self._dispatch('list_chunks', **kwargs)

    def list_algorithms(self):
        return self._dispatch('list_algorithms_impl')

    def get_algorithm_info(self, algo_id: str):
        return self._dispatch('get_algorithm_info_impl', algo_id)

    def create_kb(self, kb_id: str, display_name: Optional[str] = None, description: Optional[str] = None,
                  owner_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                  algo_id: str = '__default__'):
        return self._dispatch('create_kb_by_id', kb_id, display_name, description, owner_id, meta, algo_id)

    def update_kb(self, kb_id: str, request: KbUpdateRequest):
        return self._dispatch('update_kb_by_id', kb_id, request)

    def batch_get_kbs(self, kb_ids: List[str]):
        return self._dispatch('batch_get_kbs', KbBatchQueryRequest(kb_ids=kb_ids))

    def delete_kb(self, kb_id: str):
        return self._dispatch('delete_kb', kb_id)

    def delete_kbs(self, kb_ids: List[str]):
        return self._dispatch('delete_kbs_impl', kb_ids)

    def ensure_kb_registered(self, kb_id: str, algo_id: Optional[str] = None):
        '''Ensure the knowledge base row and algorithm binding exist in the doc service.'''
        return self._dispatch('ensure_kb_registered', kb_id, algo_id)

    def enable_scanning(self):
        '''Trigger dataset scanning for a local doc service after registrations are ready.'''
        return self._dispatch('enable_scanning')
