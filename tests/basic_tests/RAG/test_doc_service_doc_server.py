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
