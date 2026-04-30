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
from datetime import datetime

from .base import (
    AddFileItem,
    AddRequest,
    AlgorithmInfoRequest,
    CallbackEventType,
    DeleteRequest,
    DocServiceError,
    DocStatus,
    DOCUMENTS_TABLE_INFO,
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
from .utils import from_json, sha256_file, to_json

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
            # Release the DB engine so callers can clean up the backing directory
            # (sqlite on Windows keeps an exclusive handle until dispose()).
            if self._manager is not None:
                try:
                    self._manager.close()
                except Exception:
                    pass
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

        def _gen_unique_upload_path(
            self, filename: str, reserved_paths: Optional[set] = None,
            *, base_dir: Optional[str] = None,
        ):
            safe_name = os.path.basename(filename) or 'upload.bin'
            target_dir = base_dir or self._storage_dir
            file_path = os.path.join(target_dir, safe_name)
            reserved_paths = reserved_paths or set()
            if file_path not in reserved_paths and not os.path.exists(file_path):
                return file_path

            suffix = os.path.splitext(safe_name)[1]
            prefix = safe_name[:-len(suffix)] if suffix else safe_name
            for idx in range(1, 10000):
                candidate = os.path.join(target_dir, f'{prefix}-{idx}{suffix}')
                if candidate not in reserved_paths and not os.path.exists(candidate):
                    return candidate
            digest = hashlib.sha256(safe_name.encode()).hexdigest()[:8]
            return os.path.join(target_dir, f'{prefix}-{digest}{suffix}')

        @staticmethod
        async def _save_upload_file(upload_file: 'fastapi.UploadFile', file_path: str):
            with open(file_path, 'wb') as fh:
                while True:
                    chunk = await upload_file.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            await upload_file.close()

        async def _persist_uploads(
            self, files: List[fastapi.UploadFile], *,
            override: bool = False, sub_dir: str = '',
        ):
            saved_paths = []
            file_identities = []
            reserved_paths: Set[str] = set()
            target_dir = os.path.join(self._storage_dir, sub_dir) if sub_dir else self._storage_dir
            if sub_dir:
                os.makedirs(target_dir, exist_ok=True)
            for upload_file in files:
                filename = getattr(upload_file, 'filename', None) or 'upload.bin'
                if override:
                    # Legacy /upload_files behavior: write to
                    # ``storage_dir[/sub_dir]/<name>``, overwriting any
                    # existing file. ``DocManager.upload()`` derives
                    # ``doc_id`` from ``file_path``, so this lets a
                    # re-upload replace the existing document instead of
                    # creating a new one.
                    safe_name = os.path.basename(filename) or 'upload.bin'
                    file_path = os.path.join(target_dir, safe_name)
                    if file_path in reserved_paths:
                        # Two uploads in the same request collided on the same name;
                        # keep the unique-path fallback so we don't lose one of them.
                        file_path = self._gen_unique_upload_path(
                            filename, reserved_paths, base_dir=target_dir,
                        )
                else:
                    file_path = self._gen_unique_upload_path(
                        filename, reserved_paths, base_dir=target_dir,
                    )
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

        # Reserved metadata keys that the legacy /upload_files rejected; if a
        # client smuggles them into ``metadatas`` they would silently overwrite
        # internal docid/path tracking via ``parsing_service/impl.py``'s
        # ``setdefault`` calls and break later filter/delete/reparse flows.
        _LEGACY_RESERVED_META_KEYS = frozenset({'docid', 'doc_id', 'lazyllm_doc_path'})

        @classmethod
        def _parse_legacy_metadatas(cls, metadatas: Optional[str], expected_len: int) -> List[Dict[str, Any]]:
            '''Validate and parse the legacy ``metadatas`` query param.

            Returns the parsed list (empty when ``metadatas`` is falsy). Raises
            HTTPException(400) on JSON / shape / length / reserved-key issues so
            the caller doesn't have to repeat each guard inline.
            '''
            if not metadatas:
                return []
            try:
                parsed = json.loads(metadatas) or []
            except (ValueError, TypeError) as exc:
                raise fastapi.HTTPException(
                    status_code=400, detail=f'metadatas must be valid JSON: {exc}',
                )
            if not isinstance(parsed, list):
                raise fastapi.HTTPException(
                    status_code=400, detail='metadatas must be a JSON array',
                )
            # Legacy contract: rejects arrays with the wrong length or non-dict
            # entries with 400, instead of silently dropping/padding (which
            # would attach the wrong metadata to uploads) or letting AddFileItem
            # raise a 500.
            if len(parsed) != expected_len:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail=f'metadatas length {len(parsed)} does not match files length {expected_len}',
                )
            for entry in parsed:
                if not isinstance(entry, dict):
                    raise fastapi.HTTPException(
                        status_code=400, detail='each metadatas entry must be a JSON object',
                    )
                bad = cls._LEGACY_RESERVED_META_KEYS.intersection(entry.keys())
                if bad:
                    raise fastapi.HTTPException(
                        status_code=400,
                        detail=f'metadatas contains reserved keys: {sorted(bad)}',
                    )
            return parsed

        @staticmethod
        def _normalize_legacy_user_path(user_path: Optional[str]) -> str:
            '''Validate and normalize the legacy ``user_path`` query param.

            Rejects absolute paths or any value that climbs out of
            ``storage_dir`` (``..``, ``../x``, ``..\\x`` on Windows). Returns
            the relative subdirectory string ('' when no user_path).
            '''
            if not user_path:
                return ''
            if os.path.isabs(user_path):
                raise fastapi.HTTPException(
                    status_code=400, detail=f'invalid user_path: {user_path!r}',
                )
            normalized = os.path.normpath(user_path)
            climbed_out = (
                normalized in ('.', '..')
                or normalized.startswith('../')
                or normalized.startswith('..' + os.sep)
                or os.path.isabs(normalized)
            )
            if climbed_out:
                raise fastapi.HTTPException(
                    status_code=400, detail=f'invalid user_path: {user_path!r}',
                )
            sub_dir = normalized.strip('/').strip(os.sep)
            if not sub_dir:
                raise fastapi.HTTPException(
                    status_code=400, detail=f'invalid user_path: {user_path!r}',
                )
            return sub_dir

        async def _legacy_upload(
            self,
            files: List['fastapi.UploadFile'],
            override: bool,
            metadatas: Optional[str],
            group_name: Optional[str],
            user_path: Optional[str],
            *,
            response_shape: str = 'ids_and_results',  # 'ids_and_results' or 'ids_only'
        ):
            '''Shared implementation for legacy DocManager-style upload endpoints.

            Kept so DocWebModule and external callers from before the doc_service
            refactor (which used /upload_files and /add_files_to_group on the old
            ``ServerModule(DocManager(...))``) keep working against the new
            DocServer. New callers should target the /v1/docs/* surface instead.

            Compatibility behaviors preserved here:
            - ``override=True`` writes files at deterministic paths
              (``storage_dir/<user_path>/<filename>``) so DocManager.upload
              derives the same ``doc_id`` and reparses the existing document
              rather than creating a duplicate.
            - ``user_path`` namespaces uploads under a subdirectory, so two
              callers can post the same filename without colliding.
            - ``algo_id`` mirrors the kb-binding convention used by
              ``DocServer._Impl.ensure_kb_registered`` (``algo_id == kb_id``)
              so non-default groups don't get rejected by the algorithm
              validator.
            - ``metadatas`` rejects reserved internal keys (``docid``,
              ``doc_id``, ``lazyllm_doc_path``) instead of silently shadowing
              them downstream, and 400s on length / element-type mismatch
              instead of mis-attaching metadata or 500-ing inside AddFileItem.
            - Response body matches the legacy shape: ``data=[ids, results]``
              for ``/upload_files``; flat ``data=ids`` for ``/add_files_to_group``.
              ``results`` propagates per-item ``error_code`` (or ``'ok'``) from
              ``DocManager.upload``, so synchronous failures (e.g. parser
              outage -> ``PARSER_SUBMIT_FAILED``) don't masquerade as success.

            Override→reparse routing: when ``override=True`` and a doc already
            exists at the destination path in the kb (excluding FAILED /
            CANCELED, which still take the upload-retry path so caller
            metadata is applied), the shim sends those items through
            ``DocManager.reparse`` instead of ``upload``, so the legacy
            "replace + reparse" workflow doesn't get rejected by the new
            ``_assert_action_allowed(..., 'upload')`` 409 on SUCCESS docs.
            New paths in the same request still flow through ``upload``.
            Caller-supplied metadata for existing docs is merged directly
            into the documents row before reparse via
            ``_legacy_apply_metadata_to_existing_docs`` -- this avoids the
            ``patch_metadata``→reparse race that would orphan a
            DOC_UPDATE_META task and intermittently 409 the reparse.

            Known compat gaps tracked in #1090 (not exercised by the migrated
            tests so the PR ships as-is; future PRs should harden these):
            - The new file bytes are written before the doc-service validation
              runs; a 4xx from ``upload()``/``reparse()`` after override leaves
              the new bytes on disk while DB attrs still describe the old
              file. New code should prefer the staged ``/v1/docs/upload``.
            - The override+metadata write happens before reparse validation,
              so a reparse rejection (e.g. WORKING/DELETING state) commits
              the metadata change without rolling back. Mitigated for
              FAILED/CANCELED docs by routing them through upload instead.
            - The override+reparse path doesn't refresh ``content_hash`` /
              ``size_bytes`` / ``file_type`` for the replaced file —
              ``list_docs`` / ``get_doc_detail`` will continue showing the
              old file's attrs until a separate metadata patch lands.
            - Validation errors raise ``HTTPException`` and surface FastAPI's
              ``{"detail": ...}`` body, not the standard ``{code,msg,data}``
              envelope.
            '''
            self._lazy_init()
            if not files:
                raise fastapi.HTTPException(status_code=400, detail='files is required')
            kb_id = group_name or '__default__'
            # Match Document._Manager / ensure_kb_registered: each kb is bound
            # to an algorithm of the same name. Hard-coding '__default__' here
            # would 400 every non-default-group upload at validation time.
            algo_id = kb_id
            parsed_metadatas = self._parse_legacy_metadatas(metadatas, len(files))
            sub_dir = self._normalize_legacy_user_path(user_path)
            saved_paths, file_identities = await self._persist_uploads(
                files, override=override, sub_dir=sub_dir,
            )
            return self._run(lambda: self._legacy_dispatch_uploads(
                saved_paths=saved_paths,
                file_identities=file_identities,
                metadatas=parsed_metadatas,
                kb_id=kb_id,
                algo_id=algo_id,
                override=override,
                response_shape=response_shape,
            ))

        def _legacy_apply_metadata_to_existing_docs(self, doc_id_meta_pairs):
            '''Synchronously merge new metadata into the documents table for the
            given (doc_id, metadata) pairs. Used by the legacy override path
            instead of ``DocManager.patch_metadata`` so we don't enqueue a
            DOC_UPDATE_META task that would race the subsequent reparse.

            We MERGE rather than replace, matching the legacy "metadatas
            updates the named keys" semantic — callers that send a partial
            metadata dict shouldn't lose existing keys.
            '''
            if not doc_id_meta_pairs:
                return
            db = self._manager._db_manager
            Doc = db.get_table_orm_class(DOCUMENTS_TABLE_INFO['name'])
            with db.get_session() as session:
                for doc_id, patch in doc_id_meta_pairs:
                    row = session.query(Doc).filter(Doc.doc_id == doc_id).first()
                    if row is None:
                        continue
                    existing = from_json(row.meta) if row.meta else {}
                    existing.update(patch)
                    row.meta = to_json(existing)
                    row.updated_at = datetime.now()
                    session.add(row)

        def _legacy_dispatch_uploads(
            self, *, saved_paths, file_identities, metadatas, kb_id, algo_id,
            override, response_shape,
        ):
            '''Split saved files into "new doc -> upload" and (override only)
            "existing doc -> reparse + metadata patch", then merge the per-item
            results in input order so the legacy response shape stays correct.

            ``DocManager.upload`` rejects an already-SUCCESS doc with 409 (via
            ``_assert_action_allowed``), so a re-upload of the same file path
            with ``override=True`` would otherwise break the legacy
            "replace + reparse" workflow. We look up existing doc_ids by path
            and route those through ``DocManager.reparse``, then ``patch_metadata``
            for any updated metadata payload.
            '''
            # Pair each saved path with its metadata + the file_identity used
            # to build the idempotency payload.
            inputs = list(zip(saved_paths, metadatas + [{}] * (len(saved_paths) - len(metadatas)), file_identities))
            # ``exclude_failed=True`` keeps FAILED/CANCELED docs OUT of the
            # reparse pool: ``_assert_action_allowed(..., 'upload')`` already
            # accepts those states, and the upload path applies the caller's
            # fresh ``metadatas``. Reparse instead reloads metadata from the
            # existing doc row, so retrying a failed upload with new tags
            # via reparse would silently lose those tags.
            existing_by_path = (
                self._manager._list_kb_docs_by_path(kb_id, exclude_failed=True)
                if override else {}
            )
            new_inputs = [(p, m, fi) for p, m, fi in inputs if p not in existing_by_path]
            reparse_inputs = [(p, m, fi, existing_by_path[p]) for p, m, fi in inputs if p in existing_by_path]

            result_by_path = {}
            # Run reparse FIRST so its validation (``_prepare_reparse_items``
            # → ``_assert_action_allowed``) raises before any new-file upload
            # has been enqueued. If we ran upload first and then reparse
            # raised on a WORKING/DELETING existing doc, the new-file uploads
            # would already be committed -- a partial-commit failure mode.
            if reparse_inputs:
                # Apply the caller's metadata directly to the documents row
                # so the reparse worker (which reloads ``doc.meta`` in
                # ``_prepare_reparse_items``) picks up the new values.
                # We deliberately do NOT call ``DocManager.patch_metadata``
                # here: it enqueues a separate DOC_UPDATE_META task that
                # races ``reparse`` -- on a fast parser the metadata task's
                # START callback flips the doc to WORKING and the subsequent
                # ``reparse()`` 409s via ``_assert_action_allowed``, and
                # even when the race doesn't fire the metadata task is
                # orphaned in WAITING because ``reparse`` overwrites the
                # snapshot's ``current_task_id``. A direct row update
                # avoids both pitfalls while preserving the legacy
                # "overwrite refreshes tags" semantic.
                self._legacy_apply_metadata_to_existing_docs(
                    [(eid, m) for _, m, _, eid in reparse_inputs if m]
                )
                reparse_request = ReparseRequest(
                    doc_ids=[eid for _, _, _, eid in reparse_inputs],
                    kb_id=kb_id, algo_id=algo_id,
                )
                task_ids = self._manager.reparse(reparse_request)
                for (p, _, _, eid), task_id in zip(reparse_inputs, task_ids):
                    result_by_path[p] = {
                        'doc_id': eid, 'task_id': task_id,
                        'accepted': True, 'error_code': None,
                    }

            if new_inputs:
                upload_request = UploadRequest(
                    items=[AddFileItem(file_path=p, metadata=m) for p, m, _ in new_inputs],
                    kb_id=kb_id, algo_id=algo_id, source_type=SourceType.API,
                )
                upload_payload = self._build_upload_payload(
                    upload_request, [fi for _, _, fi in new_inputs],
                )
                upload_result = self._manager.run_idempotent(
                    '/v1/docs/upload', upload_request.idempotency_key, upload_payload,
                    lambda: self._manager.upload(upload_request),
                )
                if isinstance(upload_result, dict) and 'items' in upload_result:
                    upload_items = upload_result['items']
                else:
                    upload_items = upload_result or []
                for (p, _, _), item in zip(new_inputs, upload_items):
                    result_by_path[p] = item

            ordered = [result_by_path.get(p, {'doc_id': None, 'accepted': False,
                                              'error_code': 'MISSING'})
                       for p in saved_paths]
            doc_ids = [it.get('doc_id') for it in ordered]
            if response_shape == 'ids_and_results':
                # Propagate per-item status from DocManager.upload so callers
                # of /upload_files see synchronous failures (e.g. parser outage
                # -> accepted=False / error_code=PARSER_SUBMIT_FAILED) instead
                # of a misleading 'ok' for every file.
                results = [
                    'ok' if it.get('accepted', True) else (
                        it.get('error_code') or it.get('error_msg') or 'failed'
                    )
                    for it in ordered
                ]
                return [doc_ids, results]
            return doc_ids

        @app.post('/upload_files')
        async def upload_files_legacy(
            self,
            files: List['fastapi.UploadFile'] = fastapi.File(...),  # noqa: B008
            # Match legacy DocManager default: ``False`` keeps the unique-path
            # fallback so a re-upload without ``?override=true`` does not
            # silently replace an existing document.
            override: bool = fastapi.Query(False),  # noqa: B008
            metadatas: Optional[str] = fastapi.Query(None),  # noqa: B008
            group_name: Optional[str] = fastapi.Query(None),  # noqa: B008
            user_path: Optional[str] = fastapi.Query(None),  # noqa: B008
        ):
            return await self._legacy_upload(
                files, override, metadatas, group_name, user_path,
                response_shape='ids_and_results',
            )

        @app.post('/add_files_to_group')
        async def add_files_to_group_legacy(
            self,
            files: List['fastapi.UploadFile'] = fastapi.File(...),  # noqa: B008
            group_name: str = fastapi.Query(...),  # noqa: B008
            # Same legacy default as /upload_files; explicit opt-in required
            # to overwrite.
            override: bool = fastapi.Query(False),  # noqa: B008
            metadatas: Optional[str] = fastapi.Query(None),  # noqa: B008
            user_path: Optional[str] = fastapi.Query(None),  # noqa: B008
        ):
            return await self._legacy_upload(
                files, override, metadatas, group_name, user_path,
                response_shape='ids_only',
            )

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
            # DocServer is a lightweight HTTP front-end for doc CRUD; never needs
            # GPU. Default to EmptyLauncher when the caller did not pass one so we
            # don't inherit LAZYLLM_DEFAULT_LAUNCHER (e.g. 'sco' in CI) and try
            # to submit srun jobs for what should be a local Python subprocess.
            import lazyllm as _lazyllm
            effective_launcher = launcher if launcher is not None else _lazyllm.launchers.empty(sync=False)
            self._impl = ServerModule(
                self._raw_impl, port=port, launcher=effective_launcher, pythonpath=pythonpath,
            )

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
