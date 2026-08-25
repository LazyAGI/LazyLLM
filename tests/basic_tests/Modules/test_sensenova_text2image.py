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
        batch_size=1,
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


def test_u15_lite_rejects_multiple_images():
    with pytest.raises(ValueError, match='only supports n=1'):
        _module()._forward(
            input='一只猫',
            model='sensenova-u1.5-lite',
            batch_size=2,
        )
