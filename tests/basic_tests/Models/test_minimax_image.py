import base64
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths
from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxText2Image


PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC'


def response(body, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    return result


@pytest.mark.parametrize('model', MinimaxText2Image.H3_MODELS)
def test_h3_returns_successful_files_from_mixed_batch(monkeypatch, model):
    post = Mock(return_value=response({'data': [
        {'error': {'code': '1027', 'message': 'output image failed content safety check'}},
        {'b64_json': PNG, 'size': '1024x1024', 'revised_prompt': 'a red apple'},
    ]}))
    monkeypatch.setattr(requests, 'post', post)
    image = lazyllm.OnlineModule(model, source='minimax', api_key='test-key', url='https://api.minimax.io/v1')
    result = image('an apple', response_format='b64_json', size='1024x1024', n=2)
    paths = decode_query_with_filepaths(result)['files']
    assert [Path(path).read_bytes() for path in paths] == [base64.b64decode(PNG)]
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args == ('https://api.minimax.io/v1/images/generations',)
    assert kwargs['headers']['Authorization'] == 'Bearer test-key'
    assert kwargs['json'] == {'model': model, 'prompt': 'an apple', 'response_format': 'b64_json',
                              'size': '1024x1024', 'n': 2}
    assert kwargs['timeout'] >= 600
    assert kwargs['allow_redirects'] is False


def test_h3_url_and_per_call_model_override(monkeypatch):
    post = Mock(return_value=response({'data': [{'url': 'https://images.example/apple.png'}]}))
    get = Mock(return_value=response({}))
    get.return_value._content = base64.b64decode(PNG)
    get.return_value.headers['Content-Type'] = 'image/png'
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(MinimaxText2Image, '_is_internal_address', lambda self, hostname: False)
    image = lazyllm.OnlineModule(source='minimax', type='text2image', api_key='test-key')
    result = image('an apple', model='MiniMax-H3-Image', aigc_watermark=True)
    path, = decode_query_with_filepaths(result)['files']
    assert Path(path).read_bytes() == base64.b64decode(PNG)
    assert post.call_args.args == ('https://api.minimax.cn/v1/images/generations',)
    assert post.call_args.kwargs['json']['aigc_watermark'] is True
    get.assert_called_once_with('https://images.example/apple.png', timeout=180, allow_redirects=False)


def test_h3_http_error_is_not_retried(monkeypatch):
    body = {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'rate limit exceeded'},
            'request_id': 'test-request'}
    post = Mock(return_value=response(body, status=429))
    monkeypatch.setattr(requests, 'post', post)
    image = lazyllm.OnlineModule('MiniMax-H3-Image', source='minimax', api_key='test-key')
    with pytest.raises(requests.HTTPError, match='rate limit exceeded') as error:
        image.forward('an apple')
    assert error.value.response.json()['error']['message'] == 'rate limit exceeded'
    post.assert_called_once()


def test_h3_rejects_global_watermark_before_generation(monkeypatch):
    post = Mock()
    monkeypatch.setattr(requests, 'post', post)
    image = lazyllm.OnlineModule('MiniMax-H3-Image', source='minimax', api_key='test-key',
                                 url='https://api.minimax.io/v1/')
    with pytest.raises(ValueError, match='China endpoint'):
        image.forward('an apple', aigc_watermark=True)
    post.assert_not_called()


def test_h3_timeout_is_not_retried(monkeypatch):
    post = Mock(side_effect=requests.Timeout('generation timed out'))
    monkeypatch.setattr(requests, 'post', post)
    image = lazyllm.OnlineModule('MiniMax-H3-Image', source='minimax', api_key='test-key')
    with pytest.raises(requests.Timeout):
        image.forward('an apple')
    post.assert_called_once()


def test_legacy_image_generation(monkeypatch):
    post = Mock(return_value=response({'base_resp': {'status_code': 0}, 'data': {'image_base64': [PNG]}}))
    monkeypatch.setattr(requests, 'post', post)
    image = lazyllm.OnlineModule(source='minimax', type='text2image', api_key='test-key')
    result = image('an apple', response_format='base64', seed=5)
    path, = decode_query_with_filepaths(result)['files']
    assert Path(path).read_bytes() == base64.b64decode(PNG)
    assert post.call_args.args == ('https://api.minimaxi.com/v1/image_generation',)
    assert post.call_args.kwargs['json'] == {'model': 'image-01', 'prompt': 'an apple', 'response_format': 'base64',
                                             'n': 1, 'prompt_optimizer': False, 'aigc_watermark': False, 'seed': 5}
