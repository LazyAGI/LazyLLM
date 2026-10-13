import base64
import pytest

from lazyllm.module.llms.onlinemodule.supplier import sensenova


_PNG_BYTES = b'\x89PNG\r\n\x1a\nmock-image'


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _module():
    return sensenova.SenseNovaText2Image(
        api_key='sk-test',
        model='sensenova-u1.5-lite',
    )


@pytest.mark.parametrize('base_url', [
    'https://token.sensenova.cn/v1/chat/completions/',
    'https://api.sensenova.cn/compatible-mode/v1/',
    'https://proxy.example/custom/',
])
@pytest.mark.parametrize('override', [False, True])
@pytest.mark.parametrize('editing', [False, True])
def test_image_endpoint_uses_token_plan(monkeypatch, base_url, override, editing):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured['url'] = url
        return _Response({'data': [{'b64_json': base64.b64encode(_PNG_BYTES).decode()}]})

    monkeypatch.setattr(sensenova.requests, 'post', fake_post)
    monkeypatch.setattr(sensenova, 'bytes_to_file', lambda values: ['/tmp/generated.png'])
    monkeypatch.setattr(sensenova, 'encode_query_with_filepaths', lambda query, paths: paths)
    module = sensenova.SenseNovaText2Image(
        api_key='sk-test', model='sensenova-u1.5-lite', url=None if override else base_url,
    )
    result = module._forward(
        input='一只猫', url=base_url if override else None,
        files=['https://example.com/source.png'] if editing else None,
    )
    endpoint = 'images/edits' if editing else 'images/generations'
    assert captured['url'] == f'https://token.sensenova.cn/v1/{endpoint}'
    assert result == ['/tmp/generated.png']


def test_u15_lite_payload_defaults_and_b64_response(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        encoded = base64.b64encode(_PNG_BYTES).decode()
        return _Response({'data': [{'b64_json': encoded}]})

    monkeypatch.setattr(sensenova.requests, 'post', fake_post)
    monkeypatch.setattr(sensenova, 'bytes_to_file', lambda values: ['/tmp/generated.png'])
    monkeypatch.setattr(sensenova, 'encode_query_with_filepaths', lambda query, paths: paths)

    result = _module()._forward(
        input='一只猫',
        model='sensenova-u1.5-lite',
        stream_output=False,
        priority=0,
    )

    assert result == ['/tmp/generated.png']
    assert captured['url'] == 'https://token.sensenova.cn/v1/images/generations'
    assert captured['payload'] == {
        'model': 'sensenova-u1.5-lite',
        'prompt': '一只猫',
        'size': 'auto',
        'n': 1,
        'output_format': 'png',
        'response_format': 'b64_json',
        'watermark': True,
        'prompt_extend': True,
    }


def test_u15_lite_maps_workflow_parameter_names(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured['payload'] = json
        return _Response({'data': [{'url': 'https://example.com/generated.webp'}]})

    module = _module()
    module._load_images = lambda url: [('encoded', b'webp-image')]
    monkeypatch.setattr(sensenova.requests, 'post', fake_post)
    monkeypatch.setattr(sensenova, 'bytes_to_file', lambda values: ['/tmp/generated.webp'])
    monkeypatch.setattr(sensenova, 'encode_query_with_filepaths', lambda query, paths: paths)

    module._forward(
        input='一只猫',
        model='sensenova-u1.5-lite',
        image_size='4096x4096',
        n=8,
        batch_size=4,
        output_format='webp',
        response_format='url',
        watermark=False,
        prompt_extend=False,
    )

    assert captured['payload']['size'] == '4096x4096'
    assert captured['payload']['n'] == 1
    assert captured['payload']['output_format'] == 'webp'
    assert captured['payload']['response_format'] == 'url'
    assert captured['payload']['watermark'] is False
    assert captured['payload']['prompt_extend'] is False


def test_u15_lite_uses_edits_endpoint_and_images_array(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, payload=json)
        encoded = base64.b64encode(_PNG_BYTES).decode()
        return _Response({'data': [{'b64_json': encoded}]})

    module = _module()
    module._load_images = lambda file: [('encoded-image', _PNG_BYTES)]
    monkeypatch.setattr(sensenova.requests, 'post', fake_post)
    monkeypatch.setattr(sensenova, 'bytes_to_file', lambda values: ['/tmp/generated.png'])
    monkeypatch.setattr(sensenova, 'encode_query_with_filepaths', lambda query, paths: paths)

    module._forward(
        input='把背景改成雪山',
        files=['/tmp/source.png', 'https://example.com/reference.webp'],
        model='sensenova-u1.5-lite',
    )

    assert captured['url'] == 'https://token.sensenova.cn/v1/images/edits'
    assert captured['payload']['images'] == [
        {'image_url': 'data:image/png;base64,encoded-image'},
        {'image_url': 'https://example.com/reference.webp'},
    ]
    assert 'output_format' not in captured['payload']


def test_u1_fast_keeps_generation_contract(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, payload=json)
        return _Response({'data': [{'url': 'https://example.com/generated.png'}]})

    module = _module()
    module._load_images = lambda url: [('encoded', _PNG_BYTES)]
    monkeypatch.setattr(sensenova.requests, 'post', fake_post)
    monkeypatch.setattr(sensenova, 'bytes_to_file', lambda values: ['/tmp/generated.png'])
    monkeypatch.setattr(sensenova, 'encode_query_with_filepaths', lambda query, paths: paths)

    module._forward(input='信息图', model='sensenova-u1-fast')

    assert captured['url'] == 'https://token.sensenova.cn/v1/images/generations'
    assert captured['payload'] == {
        'model': 'sensenova-u1-fast',
        'prompt': '信息图',
        'size': '2752x1536',
        'n': 1,
        'watermark': True,
    }
