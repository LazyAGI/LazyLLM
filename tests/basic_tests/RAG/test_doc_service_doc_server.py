import asyncio
import json
import os
import tempfile

import pytest
import requests
from pydantic import ValidationError

from lazyllm.thirdparty import fastapi

from lazyllm.tools.rag.doc_service.doc_server import DocServer
from lazyllm.tools.rag.doc_service.base import (
    AddFileItem,
    CallbackEventType,
    DocServiceError,
    DocStatus,
    KbUpdateRequest,
    SourceType,
    TaskCallbackRequest,
    TaskCancelRequest,
    UploadRequest,
)
from lazyllm.tools.rag.utils import BaseResponse


class _UploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        self._offset = 0

    async def read(self, size: int = -1):
        if self._offset >= len(self._content):
            return b''
        if size is None or size < 0:
            size = len(self._content) - self._offset
        start = self._offset
        end = min(len(self._content), start + size)
        self._offset = end
        return self._content[start:end]

    async def close(self):
        return None


class _FakeManager:
    def __init__(self):
        self.run_calls = []
        self.upload_request = None
        self.chunk_kwargs = None
        self.cancel_response = BaseResponse(
            code=200, msg='success', data={'task_id': 'task-1', 'cancel_status': True, 'status': 'CANCELED'}
        )

    def run_idempotent(self, endpoint, idempotency_key, payload, handler):
        self.run_calls.append({
            'endpoint': endpoint,
            'idempotency_key': idempotency_key,
            'payload': payload,
        })
        return handler()

    def upload(self, request: UploadRequest):
        self.upload_request = request
        return [
            {'doc_id': item.doc_id or 'generated-doc', 'task_id': f'task-{idx}'}
            for idx, item in enumerate(request.items)
        ]

    def cancel_task(self, task_id: str):
        if isinstance(self.cancel_response, Exception):
            raise self.cancel_response
        resp = self.cancel_response
        if isinstance(resp, BaseResponse):
            return resp
        return BaseResponse.model_validate(resp)

    def list_chunks(self, **kwargs):
        self.chunk_kwargs = kwargs
        return {'items': [{'uid': 'chunk-1'}], 'total': 1, 'page': kwargs['page'], 'page_size': kwargs['page_size']}


def _decode_response(response):
    assert isinstance(response, fastapi.responses.JSONResponse)
    return json.loads(response.body.decode())


@pytest.fixture
def server_impl():
    with tempfile.TemporaryDirectory(prefix='lazyllm_doc_server_unit_') as temp_dir:
        impl = DocServer._Impl(storage_dir=temp_dir, parser_url='http://parser.test')
        impl._manager = _FakeManager()
        impl._lazy_init = lambda: None
        yield impl


def test_build_update_kb_payload_distinguishes_omitted_and_null():
    keep_req = KbUpdateRequest(display_name='Renamed', idempotency_key='kb-update-idem')
    clear_req = KbUpdateRequest(display_name='Renamed', owner_id=None, idempotency_key='kb-update-idem')

    keep_payload = DocServer._Impl._build_update_kb_payload('kb_local_idem', keep_req)
    clear_payload = DocServer._Impl._build_update_kb_payload('kb_local_idem', clear_req)

    assert keep_payload != clear_payload
    assert keep_payload['explicit_fields'] == ['display_name', 'idempotency_key']
    assert clear_payload['explicit_fields'] == ['display_name', 'idempotency_key', 'owner_id']


def test_normalize_task_callback_supports_legacy_fields():
    callback = DocServer._Impl._normalize_task_callback({
        'task_id': 'task-1',
        'task_type': 'DOC_ADD',
        'doc_id': 'doc-1',
        'kb_id': 'kb-1',
        'algo_id': '__default__',
        'task_status': 'SUCCESS',
    })

    assert isinstance(callback, TaskCallbackRequest)
    assert callback.event_type == CallbackEventType.FINISH
    assert callback.status == DocStatus.SUCCESS
    assert callback.payload == {
        'task_type': 'DOC_ADD',
        'doc_id': 'doc-1',
        'kb_id': 'kb-1',
        'algo_id': '__default__',
    }


def test_run_wraps_doc_service_error():
    response = DocServer._Impl._response(data={'ok': True})
    assert _decode_response(response)['data'] == {'ok': True}

    impl = DocServer._Impl(storage_dir='.', parser_url='http://parser.test')
    wrapped = impl._run(lambda: (_ for _ in ()).throw(DocServiceError('E_INVALID_PARAM', 'bad req', {'x': 1})))
    body = _decode_response(wrapped)

    assert body['code'] == 400
    assert body['msg'] == 'bad req'
    assert body['data']['biz_code'] == 'E_INVALID_PARAM'
    assert body['data']['x'] == 1


def test_parser_url_returns_none_when_remote_endpoint_unavailable(monkeypatch):
    server = object.__new__(DocServer)
    server._raw_impl = None
    server._impl = type('FakeImpl', (), {'_url': 'http://127.0.0.1:19002/generate'})()

    def _raise(*args, **kwargs):
        raise requests.ConnectionError('missing endpoint')

    monkeypatch.setattr(requests, 'get', _raise)

    assert server.parser_url is None


def test_task_cancel_request_requires_task_id():
    with pytest.raises(ValidationError):
        TaskCancelRequest.model_validate({})


def test_cancel_task_http_maps_conflict(server_impl):
    server_impl._manager.cancel_response = BaseResponse(
        code=409,
        msg='task cannot be canceled',
        data={'task_id': 'task-1', 'cancel_status': False, 'status': 'WORKING'},
    )

    response = server_impl.cancel_task(TaskCancelRequest(task_id='task-1', idempotency_key='idem'))
    body = _decode_response(response)

    assert server_impl._manager.run_calls[0]['endpoint'] == '/v1/tasks/cancel'
    assert server_impl._manager.run_calls[0]['idempotency_key'] == 'idem'
    assert body['code'] == 409
    assert body['data']['biz_code'] == 'E_STATE_CONFLICT'
    assert body['data']['status'] == 'WORKING'


def test_list_chunks_forwards_pagination_to_manager(server_impl):
    response = server_impl.list_chunks(
        kb_id='kb-1',
        doc_id='doc-1',
        group='block',
        algo_id='algo-1',
        page=3,
        page_size=4,
        offset=8,
    )
    body = _decode_response(response)

    assert server_impl._manager.chunk_kwargs == {
        'kb_id': 'kb-1',
        'doc_id': 'doc-1',
        'group': 'block',
        'algo_id': 'algo-1',
        'page': 3,
        'page_size': 4,
        'offset': 8,
    }
    assert body['data']['items'] == [{'uid': 'chunk-1'}]
    assert body['data']['total'] == 1


def test_upload_request_uses_idempotency_payload(server_impl):
    file_path = os.path.join(server_impl._storage_dir, 'seed.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('seed content')
    request = UploadRequest(
        kb_id='kb-upload',
        algo_id='__default__',
        source_type=SourceType.API,
        idempotency_key='upload-idem',
        items=[AddFileItem(file_path=file_path, doc_id='doc-seed')],
    )

    response = server_impl.upload_request(request)
    body = _decode_response(response)

    assert server_impl._manager.run_calls[0]['endpoint'] == '/v1/docs/upload'
    assert server_impl._manager.run_calls[0]['payload']['items'][0]['doc_id'] == 'doc-seed'
    assert body['data']['items'][0]['doc_id'] == 'doc-seed'


def test_upload_http_saves_unique_files_and_only_first_doc_id(server_impl):
    files = [
        _UploadFile('dup.txt', b'first content'),
        _UploadFile('dup.txt', b'second content'),
    ]

    response = asyncio.run(server_impl.upload(
        files=files,
        kb_id='kb-upload',
        algo_id='__default__',
        source_type=SourceType.API,
        doc_id='doc-first',
        idempotency_key='upload-http-idem',
    ))
    body = _decode_response(response)
    upload_request = server_impl._manager.upload_request

    assert body['code'] == 200
    assert len(upload_request.items) == 2
    assert upload_request.items[0].doc_id == 'doc-first'
    assert upload_request.items[1].doc_id is None
    assert upload_request.items[0].file_path != upload_request.items[1].file_path
    assert os.path.exists(upload_request.items[0].file_path)
    assert os.path.exists(upload_request.items[1].file_path)


def test_upload_files_legacy_returns_legacy_response_shape(server_impl):
    '''Regression: legacy /upload_files must accept JSON ``metadatas``, route
    to the new UploadRequest with per-item metadata, and return the legacy
    ``data=[doc_ids, results]`` body shape so DocWebModule and external
    callers (examples/rag_milvus_store flow) keep working.'''
    files = [
        _UploadFile('a.txt', b'one'),
        _UploadFile('b.txt', b'two'),
    ]
    metas = [{'comment': 'comment1'}, {'signature': 'signature2'}]

    response = asyncio.run(server_impl.upload_files_legacy(
        files=files,
        override=True,
        metadatas=json.dumps(metas),
        group_name=None,
        user_path=None,
    ))
    body = _decode_response(response)
    upload_request = server_impl._manager.upload_request

    assert body['code'] == 200
    assert upload_request.kb_id == '__default__'
    assert upload_request.algo_id == '__default__'
    assert upload_request.source_type == SourceType.API
    assert [item.metadata for item in upload_request.items] == metas
    # Legacy response shape: data = [doc_ids, results]
    assert isinstance(body['data'], list) and len(body['data']) == 2
    doc_ids, results = body['data']
    assert isinstance(doc_ids, list) and len(doc_ids) == len(files)
    assert isinstance(results, list) and len(results) == len(files)


def test_add_files_to_group_legacy_returns_flat_id_list(server_impl):
    '''Regression: /add_files_to_group must return flat doc_id list (legacy
    contract), not the new ``{items: ...}`` shape.'''
    files = [_UploadFile('c.txt', b'three'), _UploadFile('d.txt', b'four')]
    metas = [{'department': 'dpt_123'}, {'department': 'dpt_456'}]

    response = asyncio.run(server_impl.add_files_to_group_legacy(
        files=files,
        group_name='law_kg',
        override=True,
        metadatas=json.dumps(metas),
        user_path=None,
    ))
    body = _decode_response(response)
    upload_request = server_impl._manager.upload_request

    assert body['code'] == 200
    assert upload_request.kb_id == 'law_kg'
    assert [item.metadata for item in upload_request.items] == metas
    # Legacy response shape: data = doc_ids (flat list of strings)
    assert isinstance(body['data'], list)
    assert len(body['data']) == len(files)
    assert all(isinstance(x, str) for x in body['data'])


def test_legacy_upload_rejects_metadatas_length_mismatch(server_impl):
    '''Regression: legacy contract returns 400 when ``metadatas`` length differs
    from ``files`` instead of silently dropping/padding entries (which would
    attach the wrong metadata to uploads).'''
    files = [_UploadFile('a.txt', b'x'), _UploadFile('b.txt', b'y')]
    with pytest.raises(fastapi.HTTPException) as exc_info:
        asyncio.run(server_impl.upload_files_legacy(
            files=files,
            override=True,
            metadatas=json.dumps([{'k': 'v'}]),  # only 1 entry for 2 files
            group_name=None,
            user_path=None,
        ))
    assert exc_info.value.status_code == 400
    assert 'length' in str(exc_info.value.detail).lower()


def test_legacy_upload_rejects_non_dict_metadata_entry(server_impl):
    '''Regression: legacy returned 400 for malformed metadata entries instead
    of letting AddFileItem(metadata=non_dict) raise a 500.'''
    files = [_UploadFile('a.txt', b'x')]
    with pytest.raises(fastapi.HTTPException) as exc_info:
        asyncio.run(server_impl.upload_files_legacy(
            files=files,
            override=True,
            metadatas=json.dumps(['not-a-dict']),
            group_name=None,
            user_path=None,
        ))
    assert exc_info.value.status_code == 400


def test_legacy_upload_override_writes_to_deterministic_path(server_impl):
    '''Regression: ``/upload_files?override=true`` must overwrite the existing
    file at ``storage_dir/<name>`` so DocManager.upload reuses the same doc_id
    instead of creating a duplicate. _gen_unique_upload_path would otherwise
    pick ``name-1``, ``name-2``, etc. on each re-upload.'''
    storage_dir = server_impl._storage_dir
    files1 = [_UploadFile('dup.txt', b'first content')]
    asyncio.run(server_impl.upload_files_legacy(
        files=files1, override=True, metadatas=None, group_name=None, user_path=None,
    ))
    first_path = server_impl._manager.upload_request.items[0].file_path

    files2 = [_UploadFile('dup.txt', b'second content')]
    asyncio.run(server_impl.upload_files_legacy(
        files=files2, override=True, metadatas=None, group_name=None, user_path=None,
    ))
    second_path = server_impl._manager.upload_request.items[0].file_path

    assert first_path == os.path.join(storage_dir, 'dup.txt')
    assert second_path == first_path, (
        'override=True must overwrite the same path so DocManager keeps a single doc_id'
    )
    with open(second_path, 'rb') as fh:
        assert fh.read() == b'second content'


def test_legacy_upload_without_override_keeps_unique_paths(server_impl):
    '''Regression: ``override=False`` must fall back to the unique-path logic so
    we do not silently clobber an existing file on disk.'''
    storage_dir = server_impl._storage_dir
    asyncio.run(server_impl.upload_files_legacy(
        files=[_UploadFile('keep.txt', b'first')],
        override=False, metadatas=None, group_name=None, user_path=None,
    ))
    asyncio.run(server_impl.upload_files_legacy(
        files=[_UploadFile('keep.txt', b'second')],
        override=False, metadatas=None, group_name=None, user_path=None,
    ))

    second_path = server_impl._manager.upload_request.items[0].file_path
    assert second_path != os.path.join(storage_dir, 'keep.txt'), (
        'override=False must not overwrite; expected unique fallback path'
    )
    assert second_path.startswith(os.path.join(storage_dir, 'keep'))


def test_legacy_upload_uses_kb_id_as_algo_id(server_impl):
    '''Regression: ``algo_id`` must mirror the kb-binding convention so a
    ``/add_files_to_group?group_name=law_kg`` upload doesn't get rejected by
    the algorithm validator (Document._Manager binds algo_id == kb_id via
    ensure_kb_registered).
    '''
    asyncio.run(server_impl.add_files_to_group_legacy(
        files=[_UploadFile('a.txt', b'x')],
        group_name='law_kg', override=True, metadatas=None, user_path=None,
    ))
    upload_request = server_impl._manager.upload_request
    assert upload_request.kb_id == 'law_kg'
    assert upload_request.algo_id == 'law_kg', (
        'algo_id must default to kb_id, not __default__, to match '
        'ensure_kb_registered binding'
    )


def test_legacy_upload_user_path_namespaces_files(server_impl):
    '''Regression: ``user_path`` must namespace uploads under a subdirectory so
    two callers can upload the same filename without colliding (legacy contract
    that DocWebModule and external clients depend on).'''
    asyncio.run(server_impl.upload_files_legacy(
        files=[_UploadFile('report.pdf', b'team_a')],
        override=True, metadatas=None, group_name=None, user_path='team_a',
    ))
    path_a = server_impl._manager.upload_request.items[0].file_path

    asyncio.run(server_impl.upload_files_legacy(
        files=[_UploadFile('report.pdf', b'team_b')],
        override=True, metadatas=None, group_name=None, user_path='team_b',
    ))
    path_b = server_impl._manager.upload_request.items[0].file_path

    assert path_a != path_b
    assert os.path.basename(os.path.dirname(path_a)) == 'team_a'
    assert os.path.basename(os.path.dirname(path_b)) == 'team_b'
    with open(path_a, 'rb') as fh:
        assert fh.read() == b'team_a'
    with open(path_b, 'rb') as fh:
        assert fh.read() == b'team_b'


def test_legacy_upload_rejects_traversal_user_path(server_impl):
    '''Regression: malicious ``user_path`` (../, absolute) must be rejected to
    prevent escaping ``storage_dir``.'''
    for bad in ('../escape', '/etc/passwd', '..'):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            asyncio.run(server_impl.upload_files_legacy(
                files=[_UploadFile('a.txt', b'x')],
                override=True, metadatas=None, group_name=None, user_path=bad,
            ))
        assert exc_info.value.status_code == 400


def test_legacy_upload_rejects_reserved_metadata_keys(server_impl):
    '''Regression: reserved internal keys (docid, doc_id, lazyllm_doc_path) must
    be rejected with 400 instead of silently shadowing internal tracking via
    parsing_service/impl.py's setdefault().'''
    for bad_key in ('docid', 'doc_id', 'lazyllm_doc_path'):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            asyncio.run(server_impl.upload_files_legacy(
                files=[_UploadFile('a.txt', b'x')],
                override=True,
                metadatas=json.dumps([{bad_key: 'attacker_value'}]),
                group_name=None,
                user_path=None,
            ))
        assert exc_info.value.status_code == 400, f'{bad_key} should be rejected'
        assert bad_key in str(exc_info.value.detail)


def test_legacy_upload_files_propagates_per_item_failure(server_impl):
    '''Regression: when DocManager.upload returns items with ``accepted=False``
    (e.g. parser outage -> error_code=PARSER_SUBMIT_FAILED), the legacy
    /upload_files response must surface that error in the ``results`` slot
    instead of always saying ``'ok'``. Without this, a parser outage looks
    like a successful upload to legacy clients.'''
    fake = server_impl._manager

    def upload_with_failure(request):
        return [
            {'doc_id': 'd1', 'task_id': 't1', 'accepted': True, 'error_code': None},
            {'doc_id': 'd2', 'task_id': 't2', 'accepted': False,
             'error_code': 'PARSER_SUBMIT_FAILED', 'error_msg': 'parser is down'},
        ]
    fake.upload = upload_with_failure

    files = [_UploadFile('a.txt', b'one'), _UploadFile('b.txt', b'two')]
    response = asyncio.run(server_impl.upload_files_legacy(
        files=files, override=True, metadatas=None, group_name=None, user_path=None,
    ))
    body = _decode_response(response)

    assert body['code'] == 200
    doc_ids, results = body['data']
    assert doc_ids == ['d1', 'd2']
    assert results[0] == 'ok'
    assert results[1] == 'PARSER_SUBMIT_FAILED'
