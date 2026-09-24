import json
from unittest.mock import Mock

import pytest
import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.multimodal import OnlineMultiModalModule, dynamic_multimodal_config
from lazyllm.module.llms.onlinemodule.supplier import doubao


def response(body=None, status=200, content=None):
    result = requests.Response()
    result.status_code = status
    result._content = content if content is not None else json.dumps(body).encode()
    result._content_consumed = True
    result.url = 'https://ark.example/api/v3'
    return result


@pytest.fixture
def transport(monkeypatch):
    request = Mock()
    monkeypatch.setattr(requests, 'request', request)
    monkeypatch.setattr(requests, 'get', lambda url, **kwargs: request('GET', url, **kwargs))
    monkeypatch.setattr(doubao, 'bytes_to_file', lambda data: [f'/output/{i}' for i in range(len(data))])
    return request


def test_image_public_router_preserves_config_and_reference_options(transport, monkeypatch):
    transport.side_effect = [response({'data': [{'url': 'https://cdn.example/img'}]}), response(content=b'image')]
    monkeypatch.setattr(doubao.DoubaoText2Image, '_load_images', lambda *args: [('aW1n', b'img')])
    model = OnlineMultiModalModule(source='doubao', type='image_editing', model='configured-model',
                                   api_key='configured-key', url='https://ark.example/custom/v3/')
    result = model('edit', files=['reference.png'], n=20, watermark=False, stream_output=True)
    call = transport.call_args_list[0]
    assert call.args == ('POST', 'https://ark.example/custom/v3/images/generations')
    assert call.kwargs['headers']['Authorization'] == 'Bearer configured-key'
    assert call.kwargs['json'] == {
        'model': 'configured-model', 'prompt': 'edit', 'size': '1024x1024', 'seed': -1,
        'guidance_scale': 2.5, 'watermark': False, 'response_format': 'url', 'stream': False,
        'image': ['data:image/png;base64,aW1n'], 'sequential_image_generation': 'auto',
        'sequential_image_generation_options': {'max_images': 15},
    }
    assert '/output/0' in result
    assert 'headers' not in transport.call_args_list[1].kwargs


def test_video_dynamic_router_uses_request_key_model_and_url(transport, monkeypatch):
    transport.side_effect = [response({'id': 'task-1'}), response({'status': 'queued'}),
                            response({'status': 'succeeded', 'content': {'video_url': 'https://cdn.example/video'}}),
                            response(content=b'video')]
    monkeypatch.setitem(lazyllm.globals['config'], 'doubao_api_key', 'request-key')
    model = OnlineMultiModalModule(source='dynamic', dynamic_auth=True, type='text2video')
    with dynamic_multimodal_config(model, source='doubao', model='doubao-seedance-2-5-260628',
                                   url='https://ark.example/custom/v3'):
        result = model('animate', files=['https://cdn.example/ref'], image_roles=['reference_image'],
                       resolution='720p', duration=5, ratio='1:1', watermark=False, poll_interval=0)
    create = transport.call_args_list[0]
    assert create.args == ('POST', 'https://ark.example/custom/v3/contents/generations/tasks')
    assert create.kwargs['headers']['Authorization'] == 'Bearer request-key'
    body = create.kwargs['json']
    assert body['model'] == 'doubao-seedance-2-5-260628'
    assert body['omni_reference_task_type'] == 'auto'
    assert body['content'][1]['role'] == 'reference_image'
    assert (body['resolution'], body['duration'], body['ratio'], body['watermark']) == ('720p', 5, '1:1', False)
    assert 'camera_fixed' not in body
    assert transport.call_args_list[1].args == ('GET', create.args[1] + '/task-1')
    assert '/output/0' in result


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'expired', 'unexpected'])
def test_video_terminal_states_stop_polling(transport, status):
    transport.side_effect = [response({'id': 'task-1'}),
                            response({'status': status, 'error': {'message': 'stopped'}})]
    model = doubao.DoubaoText2Video(api_key='key')
    with pytest.raises(RuntimeError, match=status):
        model.forward('animate', poll_interval=0)
    assert transport.call_count == 2


@pytest.mark.parametrize('status', [401, 429, 500])
def test_creation_errors_preserve_details_without_retry(transport, status):
    transport.return_value = response({'error': {'code': 'ProviderError', 'message': 'reason'}}, status)
    with pytest.raises(requests.HTTPError, match=f'{status}: ProviderError reason'):
        doubao.DoubaoText2Video(api_key='key').forward('animate')
    assert transport.call_count == 1


def test_timeout_after_creation_does_not_create_another_task(transport, monkeypatch):
    transport.return_value = response({'id': 'task-1'})
    monkeypatch.setattr(doubao.time, 'monotonic', Mock(side_effect=[0, 10]))
    with pytest.raises(TimeoutError, match='task-1'):
        doubao.DoubaoText2Video(api_key='key').forward('animate', timeout=5)
    assert transport.call_count == 1


@pytest.mark.parametrize('body', [{}, {'data': []}, {'data': [{'error': {'message': 'filtered'}}]}])
def test_image_empty_or_filtered_output_fails(transport, body):
    transport.return_value = response(body)
    with pytest.raises(RuntimeError):
        doubao.DoubaoText2Image(api_key='key').forward('draw')
    assert transport.call_count == 1


def test_failed_download_is_not_saved_as_media(transport):
    transport.side_effect = [response({'data': [{'url': 'https://cdn.example/expired'}]}), response({}, 403)]
    with pytest.raises(requests.HTTPError):
        doubao.DoubaoText2Image(api_key='key').forward('draw')


def test_video_first_last_frame_and_runtime_override(transport):
    transport.side_effect = [response({'id': 'task-1'}),
                            response({'status': 'succeeded', 'content': {'video_url': 'https://cdn.example/video'}}),
                            response(content=b'video')]
    model = doubao.DoubaoText2Video(api_key=['key'], model='initial')
    model.forward('animate', model='runtime-model', url='https://ark.example/override/',
                  files=['data:image/png;base64,Zmlyc3Q=', 'https://cdn.example/last'],
                  image_roles=['first_frame', 'last_frame'], camerafixed=True)
    call = transport.call_args_list[0]
    assert call.args[1] == 'https://ark.example/override/contents/generations/tasks'
    assert call.kwargs['json']['model'] == 'runtime-model'
    assert call.kwargs['json']['camera_fixed'] is True
    assert [item['role'] for item in call.kwargs['json']['content'][1:]] == ['first_frame', 'last_frame']
    assert transport.call_args_list[1].kwargs['headers'] == call.kwargs['headers']
