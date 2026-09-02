import lazyllm

from lazyllm.module.llms.onlinemodule.supplier import qwen as qwen_supplier
from lazyllm.module.llms.onlinemodule.supplier import siliconflow as siliconflow_supplier
from lazyllm.module.llms.onlinemodule.supplier.qwen import QwenText2Video
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowText2Video


class _Response:
    def __init__(self, data=None, content=b''):
        self._data = data
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_video_supplier_registration():
    assert lazyllm.online.text2video.qwen is QwenText2Video
    assert lazyllm.online.text2video.siliconflow is SiliconFlowText2Video


def test_qwen_text2video_uses_dashscope_async_api(monkeypatch):
    captured = {'get_urls': []}

    def fake_post(url, **kwargs):
        captured.update(post_url=url, post_kwargs=kwargs)
        return _Response({'output': {'task_id': 'task-123'}})

    def fake_get(url, **kwargs):
        captured['get_urls'].append(url)
        if url == 'https://video.example/result.mp4':
            return _Response(content=b'qwen-video')
        return _Response({'output': {
            'task_status': 'SUCCEEDED',
            'video_url': 'https://video.example/result.mp4',
        }})

    monkeypatch.setattr(qwen_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(qwen_supplier.requests, 'get', fake_get)
    monkeypatch.setattr(qwen_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(qwen_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = QwenText2Video(api_key='test-key', model='wan2.6-t2v')
    result = video._forward(input='a bird flying', resolution='480p', poll_interval=0)

    assert captured['post_url'] == (
        'https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis'
    )
    assert captured['post_kwargs']['headers']['X-DashScope-Async'] == 'enable'
    assert captured['post_kwargs']['json']['model'] == 'wan2.6-t2v'
    assert captured['post_kwargs']['json']['parameters']['size'] == '1280*720'
    assert captured['get_urls'] == [
        'https://dashscope.aliyuncs.com/api/v1/tasks/task-123',
        'https://video.example/result.mp4',
    ]
    assert result == [b'qwen-video']


def test_qwen_image2video_sends_first_frame(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response({'output': {'task_id': 'task-123'}})

    def fake_get(url, **kwargs):
        if url == 'https://video.example/result.mp4':
            return _Response(content=b'qwen-video')
        return _Response({'output': {
            'task_status': 'SUCCEEDED',
            'video_url': 'https://video.example/result.mp4',
        }})

    monkeypatch.setattr(qwen_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(qwen_supplier.requests, 'get', fake_get)
    monkeypatch.setattr(qwen_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(qwen_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = QwenText2Video(api_key='test-key', model='wan2.6-i2v-flash')
    video._forward(
        input='animate the cat',
        files=['https://image.example/cat.png'],
        image_roles=['first_frame'],
        resolution='720p',
        poll_interval=0,
    )

    assert captured['json']['input']['img_url'] == 'https://image.example/cat.png'
    assert captured['json']['parameters']['resolution'] == '720P'


def test_qwen_wan3_all_in_one_sends_first_and_last_frames(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response({'output': {'task_id': 'task-wan3'}})

    def fake_get(url, **kwargs):
        if url == 'https://video.example/wan3.mp4':
            return _Response(content=b'wan3-video')
        return _Response({'output': {
            'task_status': 'SUCCEEDED',
            'video_url': 'https://video.example/wan3.mp4',
        }})

    monkeypatch.setattr(qwen_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(qwen_supplier.requests, 'get', fake_get)
    monkeypatch.setattr(qwen_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(qwen_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = QwenText2Video(api_key='test-key')
    result = video._forward(
        input='transition between the two frames',
        files=['https://image.example/first.png', 'https://image.example/last.png'],
        image_roles=['first_frame', 'last_frame'],
        resolution='480p',
        ratio='adaptive',
        duration=30,
        poll_interval=0,
    )

    payload = captured['json']
    assert payload['model'] == 'wan3.0-video'
    assert payload['input']['media'] == [
        {'type': 'first_frame', 'url': 'https://image.example/first.png'},
        {'type': 'last_frame', 'url': 'https://image.example/last.png'},
    ]
    assert payload['parameters'] == {
        'duration': 30,
        'prompt_extend': True,
        'resolution': '480P',
        'ratio': 'adaptive',
        'audio': True,
    }
    assert result == [b'wan3-video']


def test_qwen_wan3_all_in_one_sends_multiple_reference_images(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response({'output': {'task_id': 'task-wan3'}})

    def fake_get(url, **kwargs):
        if url == 'https://video.example/wan3.mp4':
            return _Response(content=b'wan3-video')
        return _Response({'output': {
            'task_status': 'SUCCEEDED',
            'video_url': 'https://video.example/wan3.mp4',
        }})

    monkeypatch.setattr(qwen_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(qwen_supplier.requests, 'get', fake_get)
    monkeypatch.setattr(qwen_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(qwen_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = QwenText2Video(api_key='test-key', model='wan3.0-video-prime')
    video._forward(
        input='keep all referenced subjects consistent',
        files=['https://image.example/1.png', 'https://image.example/2.png'],
        image_roles=['reference_image', 'reference_image'],
        ratio='adaptive',
        poll_interval=0,
    )

    assert captured['json']['input']['media'] == [
        {'type': 'reference_image', 'url': 'https://image.example/1.png'},
        {'type': 'reference_image', 'url': 'https://image.example/2.png'},
    ]


def test_siliconflow_image2video_uses_submit_and_status(monkeypatch):
    captured = {'posts': []}

    def fake_post(url, **kwargs):
        captured['posts'].append((url, kwargs))
        if url.endswith('/video/submit'):
            return _Response({'requestId': 'request-123'})
        return _Response({
            'status': 'Succeed',
            'results': {'videos': [{'url': 'https://video.example/result.mp4'}]},
        })

    monkeypatch.setattr(siliconflow_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(
        siliconflow_supplier.requests,
        'get',
        lambda *args, **kwargs: _Response(content=b'siliconflow-video'),
    )
    monkeypatch.setattr(siliconflow_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(siliconflow_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = SiliconFlowText2Video(api_key='test-key')
    result = video._forward(
        input='a cat running on grass',
        files=['https://image.example/cat.png'],
        image_roles=['first_frame'],
        resolution='480p',
        poll_interval=0,
    )

    submit_url, submit_kwargs = captured['posts'][0]
    assert submit_url == 'https://api.siliconflow.cn/v1/video/submit'
    assert submit_kwargs['json'] == {
        'model': 'Wan-AI/Wan2.2-I2V-A14B',
        'prompt': 'a cat running on grass',
        'image_size': '1280x720',
        'image': 'https://image.example/cat.png',
    }
    assert captured['posts'][1][0] == 'https://api.siliconflow.cn/v1/video/status'
    assert captured['posts'][1][1]['json'] == {'requestId': 'request-123'}
    assert result == [b'siliconflow-video']
