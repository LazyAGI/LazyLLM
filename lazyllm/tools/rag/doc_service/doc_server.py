from __future__ import annotations

import hashlib
import json
import os
import traceback
from typing import Any, Dict, List, Optional

from lazyllm import LOG, FastapiApp as app, ModuleBase, ServerModule, UrlModule, once_wrapper
from lazyllm.thirdparty import fastapi

from ..utils import BaseResponse, _get_default_db_config, ensure_call_endpoint
from .base import (
    AddRequest, DeleteRequest, DocServiceError, KbBatchQueryRequest, KbCreateRequest, KbUpdateRequest,
    MetadataPatchRequest, ReparseRequest,
)
from .base import CallbackEventType, DocStatus, SourceType, TaskCallbackRequest
from .base import TransferRequest
from .base import UploadRequest, AddFileItem
from .doc_manager import DocManager
from ..parsing_service.base import TaskStatus, TaskType


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
        ):
            self._storage_dir = storage_dir
            self._db_config = db_config
            self._parser_db_config = parser_db_config
            self._parser_poll_interval = parser_poll_interval
            self._parser_url = parser_url
            self._callback_url = callback_url
            self._parser = None
            self._manager = None

        @once_wrapper(reset_on_pickle=True)
        def _lazy_init(self):
            os.makedirs(self._storage_dir, exist_ok=True)
            if not self._parser_url:
                raise ValueError('parser_url is required; doc_service no longer starts a mock parsing server')
            self._manager = DocManager(
                db_config=self._db_config,
                parser_url=self._parser_url,
                callback_url=self._callback_url,
            )

        def stop(self):
            return None

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
            items = file_identities
            if items is None:
                items = []
                for idx, item in enumerate(request.items):
                    content_hash = None
                    size_bytes = None
                    if os.path.exists(item.file_path):
                        with open(item.file_path, 'rb') as fh:
                            content = fh.read()
                        content_hash = hashlib.sha256(content).hexdigest()
                        size_bytes = len(content)
                    items.append({
                        'filename': os.path.basename(item.file_path),
                        'content_hash': content_hash,
                        'size_bytes': size_bytes,
                        'doc_id': item.doc_id if idx == 0 else None,
                    })
            return {
                'kb_id': request.kb_id,
                'algo_id': request.algo_id,
                'source_type': request.source_type.value,
                'idempotency_key': request.idempotency_key,
                'items': items,
            }

        @staticmethod
        def _build_update_kb_payload(kb_id: str, request: KbUpdateRequest):
            payload = request.model_dump(mode='json', exclude_unset=True)
            payload['kb_id'] = kb_id
            payload['explicit_fields'] = sorted(request.model_fields_set)
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
            return os.path.join(self._storage_dir, f'{prefix}-{hashlib.sha256(safe_name.encode()).hexdigest()[:8]}{suffix}')

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
            request: 'fastapi.Request',
            kb_id: str = '__default__',
            algo_id: str = '__default__',
            source_type: SourceType = SourceType.API,
            doc_id: Optional[str] = None,
            idempotency_key: Optional[str] = None,
        ):
            self._lazy_init()
            form = await request.form()
            files = form.getlist('files')
            if not files:
                raise fastapi.HTTPException(status_code=400, detail='files is required')
            buffered_files = []
            file_identities = []
            for idx, file in enumerate(files):
                filename = getattr(file, 'filename', None) or str(getattr(file, 'name', 'upload.bin'))
                content = await file.read() if hasattr(file, 'read') else file.file.read()
                buffered_files.append({'filename': filename, 'content': content})
                file_identities.append({
                    'filename': filename,
                    'content_hash': hashlib.sha256(content).hexdigest(),
                    'size_bytes': len(content),
                    'doc_id': doc_id if idx == 0 else None,
                })

            def _handle_upload():
                saved_paths = []
                reserved_paths = set()
                for item in buffered_files:
                    file_path = self._gen_unique_upload_path(item['filename'], reserved_paths)
                    with open(file_path, 'wb') as fh:
                        fh.write(item['content'])
                    saved_paths.append(file_path)
                    reserved_paths.add(file_path)
                upload_request = UploadRequest(
                    items=[AddFileItem(file_path=path, doc_id=(doc_id if idx == 0 else None))
                           for idx, path in enumerate(saved_paths)],
                    kb_id=kb_id,
                    algo_id=algo_id,
                    source_type=source_type,
                    idempotency_key=idempotency_key,
                )
                return {'items': self._manager.upload(upload_request)}

            payload = {
                'kb_id': kb_id,
                'algo_id': algo_id,
                'source_type': source_type.value,
                'idempotency_key': idempotency_key,
                'items': file_identities,
            }
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/docs/upload', idempotency_key, payload, _handle_upload
            ))

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
        async def cancel_task(self, request: 'fastapi.Request'):
            payload = await request.json()
            task_id = payload.get('task_id')
            if not task_id:
                raise fastapi.HTTPException(status_code=400, detail='task_id is required')
            idempotency_key = payload.get('idempotency_key')

            def _cancel():
                resp = self._manager.cancel_task(task_id)
                if resp.code == 404:
                    raise DocServiceError('E_NOT_FOUND', resp.msg, resp.data)
                if resp.code == 409:
                    raise DocServiceError('E_STATE_CONFLICT', resp.msg, resp.data)
                if resp.code != 200:
                    raise DocServiceError('E_INVALID_PARAM', resp.msg, resp.data)
                return resp.data
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/tasks/cancel', idempotency_key, payload, _cancel
            ))

        def task_callback(self, callback: Any):
            self._lazy_init()
            return self._run(lambda: self._manager.on_task_callback(self._normalize_task_callback(callback)))

        @app.post('/v1/internal/callbacks/tasks')
        async def task_callback_http(self, request: 'fastapi.Request'):
            self._lazy_init()
            payload = await request.json()
            return self.task_callback(payload)

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
        async def get_algorithm_info(self, request: 'fastapi.Request'):
            self._lazy_init()
            payload = await request.json()
            algo_id = payload.get('algo_id')
            if not algo_id:
                return self._response(data={'biz_code': 'E_INVALID_PARAM'}, code=400,
                                      msg='algo_id is required', status_code=400)
            return self._run(lambda: self._manager.get_algorithm_info(algo_id))

        def get_algorithm_info_impl(self, algo_id: str):
            self._lazy_init()
            return self._run(lambda: self._manager.get_algorithm_info(algo_id))

        @app.get('/v1/chunks')
        def list_chunks(self, page: int = 1, page_size: int = 20):
            self._lazy_init()
            return self._run(lambda: self._manager.list_chunks(page=page, page_size=page_size))

        @app.post('/v1/tasks/batch')
        async def get_tasks_batch(self, request: 'fastapi.Request'):
            self._lazy_init()
            payload = await request.json()
            task_ids = payload.get('task_ids') or []
            return self._run(lambda: self._manager.get_tasks_batch(task_ids))

        def get_tasks_batch_impl(self, task_ids: List[str]):
            self._lazy_init()
            return self._run(lambda: self._manager.get_tasks_batch(task_ids))

        @app.post('/v1/tasks/info')
        async def get_task_info(self, request: 'fastapi.Request'):
            self._lazy_init()
            payload = await request.json()
            task_id = payload.get('task_id')
            if not task_id:
                return self._response(data={'biz_code': 'E_INVALID_PARAM'}, code=400,
                                      msg='task_id is required', status_code=400)
            resp = self._manager.get_task(task_id)
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
        async def delete_kbs(self, request: 'fastapi.Request'):
            self._lazy_init()
            payload = await request.json()
            kb_ids = payload.get('kb_ids') or []
            idempotency_key = payload.get('idempotency_key')
            return self._run(lambda: self._manager.run_idempotent(
                '/v1/kbs:delete', idempotency_key, payload, lambda: self._manager.delete_kbs(kb_ids)
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
        launcher=None,
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
            )
            self._impl = ServerModule(self._raw_impl, port=port, launcher=launcher)

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

    def cancel_task(self, task_id: str):
        return self._dispatch('cancel_task_by_id', task_id)

    def list_kbs(self, **kwargs):
        return self._dispatch('list_kbs', **kwargs)

    def get_kb(self, kb_id: str):
        return self._dispatch('get_kb', kb_id)

    def list_chunks(self, page: int = 1, page_size: int = 20):
        return self._dispatch('list_chunks', page, page_size)

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
