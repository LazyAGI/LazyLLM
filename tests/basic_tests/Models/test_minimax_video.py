from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths
from lazyllm.module.llms.onlinemodule.map_model_type import get_model_type
from lazyllm.module.llms.onlinemodule.supplier import minimax as supplier


class _Response:
    def __init__(self, data=None, content=b'', error=None):
        self.data, self.content, self.error = data, content, error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.data


@pytest.fixture
def transport(monkeypatch, tmp_path):
    request = Mock(return_value=_Response({'task_id': 'task-123'}))
    download = Mock(return_value=_Response(content=b'\x00\x00\x00\x18ftypmp42'))
    monkeypatch.setattr(supplier.requests, 'request', request)
    monkeypatch.setattr(supplier.requests, 'get', download)
    write_files = supplier.bytes_to_file
    monkeypatch.setattr(supplier, 'bytes_to_file', lambda values: write_files(values, str(tmp_path)))
    return request, download


def _task(status='succeeded', **kwargs):
    return _Response({'task': {'status': status, 'content': {'url': 'https://video.example/result.mp4'}, **kwargs}})


def test_registration_and_model_routing():
    assert lazyllm.online.text2video.minimax is supplier.MinimaxText2Video
    assert get_model_type('MiniMax-H3') == 'text2video'
    module = lazyllm.OnlineModule('MiniMax-H3', source='minimax', api_key='test-key')
    assert isinstance(module, supplier.MinimaxText2Video)
    assert module._model_name == 'MiniMax-H3'


@pytest.mark.parametrize('region,base_url', [
    ('global', 'https://api.minimax.io/v2'), ('cn', 'https://api.minimaxi.com/v2'),
])
def test_text2video_payload_polling_and_output(transport, region, base_url):
    request, download = transport
    request.side_effect = [_Response({'task_id': 'task/123'}), _task('queued'), _task('running'), _task()]
    module = lazyllm.OnlineMultiModalModule(source='minimax', type='text2video', api_key='test-key', region=region)
    result = module('A bird in flight', duration=5, ratio='16:9', poll_interval=0,
                    callback_url='https://callback.example/video', stream_output=False, priority=1)
    create = request.call_args_list[0]
    assert create.args == ('POST', base_url + '/video_generation')
    assert create.kwargs['headers']['Authorization'] == 'Bearer test-key'
    assert create.kwargs['json'] == {
        'model': 'MiniMax-H3', 'content': [{'type': 'text', 'text': 'A bird in flight'}],
        'resolution': '2K', 'duration': 5, 'ratio': '16:9',
        'callback_url': 'https://callback.example/video',
    }
    for call in request.call_args_list[1:]:
        assert call.args == ('GET', base_url + '/query/video_generation/task%2F123')
        assert call.kwargs['headers'] == create.kwargs['headers']
        assert 0 < call.kwargs['timeout'] <= 60
    assert download.call_args.args == ('https://video.example/result.mp4',)
    assert 'headers' not in download.call_args.kwargs
    files = decode_query_with_filepaths(result)['files']
    assert len(files) == 1
    assert Path(files[0]).suffix == '.mp4'
    assert Path(files[0]).read_bytes() == b'\x00\x00\x00\x18ftypmp42'


def test_runtime_url_override_and_regional_watermark(transport):
    request, _ = transport
    request.side_effect = [_Response({'task_id': 'task-123'}), _task()]
    module = lazyllm.OnlineMultiModalModule(
        'MiniMax-H3', source='minimax', api_key='test-key', region='cn', url='https://proxy.example/v2/')
    module.forward('A bird', url='https://runtime.example/v2', aigc_watermark=False, poll_interval=0)
    assert request.call_args_list[0].args[1] == 'https://runtime.example/v2/video_generation'
    assert request.call_args_list[0].kwargs['json']['aigc_watermark'] is False
    assert module._base_url == 'https://proxy.example/v2/'


@pytest.mark.parametrize('options', [
    {'input': ''}, {'input': ' '}, {'input': 'a' * 7001}, {'duration': 3}, {'duration': 16},
    {'duration': True}, {'duration': 5.5}, {'resolution': '480p'}, {'ratio': 'adaptive'},
    {'files': ['https://image.example/first.png']}, {'model': 'unsupported'},
    {'aigc_watermark': True}, {'timeout': 0}, {'poll_interval': -1},
])
def test_invalid_request_does_not_create_task(transport, options):
    request, download = transport
    module = lazyllm.online.text2video.minimax(api_key='test-key')
    with pytest.raises(ValueError):
        module.forward(**{'input': 'A bird', **options})
    request.assert_not_called()
    download.assert_not_called()


def test_invalid_region():
    with pytest.raises(ValueError, match='region'):
        lazyllm.online.text2video.minimax(api_key='test-key', region='unsupported')


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'unknown', None])
def test_unsuccessful_task_is_not_downloaded(transport, status):
    request, download = transport
    request.side_effect = [_Response({'task_id': 'task-123'}), _task(status, error={'message': 'Task stopped'})]
    module = lazyllm.online.text2video.minimax(api_key='test-key')
    with pytest.raises(RuntimeError, match='Task stopped'):
        module.forward('A bird', poll_interval=0)
    download.assert_not_called()


@pytest.mark.parametrize('response,error_type', [
    (_Response({}), RuntimeError), (_Response([]), ValueError),
    (_Response({'error': {'message': 'Invalid request'}}), RuntimeError),
    (_Response(error=requests.HTTPError('Bad request')), requests.HTTPError),
])
def test_creation_failure_is_not_retried(transport, response, error_type):
    request, download = transport
    request.return_value = response
    module = lazyllm.online.text2video.minimax(api_key='test-key')
    with pytest.raises(error_type):
        module.forward('A bird')
    assert request.call_count == 1
    download.assert_not_called()


def test_success_without_url_is_rejected(transport):
    request, download = transport
    request.side_effect = [_Response({'task_id': 'task-123'}), _task(content={})]
    module = lazyllm.online.text2video.minimax(api_key='test-key')
    with pytest.raises(RuntimeError, match='without a video URL'):
        module.forward('A bird', poll_interval=0)
    download.assert_not_called()


def test_polling_obeys_deadline(transport, monkeypatch):
    request, download = transport
    request.side_effect = [_Response({'task_id': 'task-123'}), _task('running')]
    monkeypatch.setattr(supplier.time, 'monotonic', Mock(side_effect=[0, 1, 2, 4]))
    sleep = Mock()
    monkeypatch.setattr(supplier.time, 'sleep', sleep)
    module = lazyllm.online.text2video.minimax(api_key='test-key')
    with pytest.raises(TimeoutError, match='task-123'):
        module.forward('A bird', timeout=3, poll_interval=5)
    assert request.call_args.kwargs['timeout'] == 2
    sleep.assert_called_once_with(1)
    download.assert_not_called()


@pytest.mark.parametrize('stage', ['create', 'poll', 'download'])
def test_transport_errors_are_propagated_without_recreating_task(transport, stage):
    request, download = transport
    responses = [_Response({'task_id': 'task-123'}), _task()]
    error = requests.Timeout('Request timed out')
    if stage == 'create':
        responses[0] = error
    elif stage == 'poll':
        responses[1] = error
    else:
        download.side_effect = error
    request.side_effect = responses
    module = lazyllm.online.text2video.minimax(api_key='test-key')
    with pytest.raises(requests.Timeout, match='Request timed out'):
        module.forward('A bird', poll_interval=0)
    assert sum(call.args[0] == 'POST' for call in request.call_args_list) == 1
