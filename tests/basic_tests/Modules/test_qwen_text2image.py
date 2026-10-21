import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import requests

import lazyllm
from lazyllm.common.globals import new_session
from lazyllm.module.llms.model_config_inject import inject_model_config
from lazyllm.module.llms.onlinemodule.supplier import qwen


_MODELS = [
    ('qwen-image-3.0-pro', 'multimodal-generation/generation'),
    ('wan2.6-image', 'multimodal-generation/generation'),
    ('wan2.7-image', 'text2image/image-synthesis'),
]


@pytest.fixture
def image_requests(monkeypatch):
    pytest.importorskip('dashscope')
    captured = []

    def send(session, request, **kwargs):
        captured.append(request)
        output = {'choices': [{'message': {
            'role': 'assistant', 'content': [{'image': 'https://images.example/result.png'}],
        }}]}
        if request.method == 'GET':
            output = {'task_status': 'SUCCEEDED', 'results': [{'url': 'https://images.example/result.png'}]}
        elif request.url.endswith('/image-synthesis'):
            output = {'task_id': 'task-123', 'task_status': 'PENDING'}
        response = requests.Response()
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        response._content = json.dumps({'request_id': 'test-request', 'output': output, 'usage': {}}).encode()
        response.request = request
        return response

    monkeypatch.setattr(requests.Session, 'send', send)
    monkeypatch.setattr(qwen.QwenText2Image, '_load_images', lambda self, paths: [('encoded', b'image')])
    monkeypatch.setattr(qwen, 'bytes_to_file', lambda values: ['/tmp/generated.png'])
    monkeypatch.setattr(qwen, 'encode_query_with_filepaths', lambda query, paths: paths)
    return captured


@pytest.mark.parametrize('model,endpoint', _MODELS)
@pytest.mark.parametrize('base,root', [
    ('https://proxy.example', 'https://proxy.example/api/v1'),
    ('https://proxy.example/', 'https://proxy.example/api/v1'),
    ('https://proxy.example/prefix/api/v1', 'https://proxy.example/prefix/api/v1'),
    ('https://proxy.example/prefix/api/v1/', 'https://proxy.example/prefix/api/v1'),
])
@pytest.mark.parametrize('entry', ['constructor', 'constructor_url', 'runtime', 'runtime_base_url', 'dynamic'])
def test_image_requests_use_configured_endpoint(image_requests, model, endpoint, base, root, entry):
    with new_session():
        if entry == 'dynamic':
            module = lazyllm.OnlineMultiModalModule(source='dynamic', type='text2image')
            module.name = 'image_generator'
            inject_model_config({'image_generator': {
                'type': 'text2image', 'source': 'qwen', 'model': model,
                'base_url': base, 'api_key': 'test-key',
            }})
            result = module('a cat')
        else:
            init_kwargs, call_kwargs = {'base_url': 'https://unused.example'}, {}
            if entry.startswith('constructor'):
                init_kwargs = {'base_url' if entry == 'constructor' else 'url': base}
            else:
                call_kwargs = {'url' if entry == 'runtime' else 'base_url': base}
            module = qwen.QwenText2Image(model=model, api_key='test-key', **init_kwargs)
            result = module('a cat', **call_kwargs)

    assert result == ['/tmp/generated.png']
    expected = [f'{root}/services/aigc/{endpoint}']
    if endpoint.endswith('/image-synthesis'):
        expected.append(f'{root}/tasks/task-123')
    assert [request.url for request in image_requests] == expected
    assert all(request.headers['Authorization'] == 'Bearer test-key' for request in image_requests)
    assert 'base_address' not in json.loads(image_requests[0].body).get('parameters', {})


@pytest.mark.parametrize('sdk_root', ['https://dashscope.aliyuncs.com/api/v1', 'https://global.example/api/v1'])
def test_image_default_keeps_sdk_endpoint(image_requests, monkeypatch, sdk_root):
    qwen._ensure_dashscope_urls_initialized()
    monkeypatch.setattr(qwen.dashscope, 'base_http_api_url', sdk_root)
    module = qwen.QwenText2Image(model='qwen-image-plus', api_key='test-key')
    module('a cat', url='https://override.example')
    module('a cat')
    assert [request.url for request in image_requests] == [
        'https://override.example/api/v1/services/aigc/multimodal-generation/generation',
        f'{sdk_root}/services/aigc/multimodal-generation/generation',
    ]
    assert qwen.dashscope.base_http_api_url == sdk_root


def test_image_editing_uses_configured_endpoint(image_requests):
    module = qwen.QwenText2Image(
        model='qwen-image-edit-plus', api_key='test-key', type='image_editing', base_url='https://edit.example',
    )
    assert module('make it blue', files=['/tmp/input.png']) == ['/tmp/generated.png']
    assert image_requests[0].url == 'https://edit.example/api/v1/services/aigc/multimodal-generation/generation'


def test_image_endpoints_are_isolated_between_concurrent_modules(image_requests, monkeypatch):
    first = qwen.QwenText2Image(model='qwen-image-plus', api_key='first-key', base_url='https://first.example')
    second = qwen.QwenText2Image(model='qwen-image-plus', api_key='second-key', base_url='https://second.example')
    original_send = requests.Session.send
    barrier = Barrier(2)

    def send(session, request, **kwargs):
        barrier.wait(timeout=10)
        return original_send(session, request, **kwargs)

    monkeypatch.setattr(requests.Session, 'send', send)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda module: module('a cat'), [first, second])) == [
            ['/tmp/generated.png'], ['/tmp/generated.png'],
        ]
    assert {(request.url, request.headers['Authorization']) for request in image_requests} == {
        ('https://first.example/api/v1/services/aigc/multimodal-generation/generation', 'Bearer first-key'),
        ('https://second.example/api/v1/services/aigc/multimodal-generation/generation', 'Bearer second-key'),
    }
