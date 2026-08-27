import base64

import lazyllm

from lazyllm.module.llms.onlinemodule.supplier import openrouter as openrouter_supplier

from lazyllm.module.llms.onlinemodule.supplier.openrouter import (
    OpenRouterChat,
    OpenRouterEmbed,
    OpenRouterSTT,
    OpenRouterTTS,
    OpenRouterText2Image,
    OpenRouterText2Video,
)


def test_openrouter_chat_defaults_and_registration():
    module = OpenRouterChat(api_key='sk-or-test', stream=False)

    assert module.series == 'openrouter'
    assert module._base_url == 'https://openrouter.ai/api/v1/'
    assert module._model_name == 'openrouter/free'
    assert lazyllm.online.chat.openrouter is OpenRouterChat


def test_openrouter_chat_accepts_catalog_model_ids():
    module = lazyllm.OnlineModule(
        model='openrouter/free',
        source='openrouter',
        url='https://openrouter.ai/api/v1/',
        api_key='sk-or-test',
    )

    assert isinstance(module, OpenRouterChat)
    assert module._model_name == 'openrouter/free'


def test_openrouter_multimodal_registration():
    assert lazyllm.online.embed.openrouter is OpenRouterEmbed
    assert lazyllm.online.text2image.openrouter is OpenRouterText2Image
    assert lazyllm.online.text2video.openrouter is OpenRouterText2Video
    assert lazyllm.online.tts.openrouter is OpenRouterTTS
    assert lazyllm.online.stt.openrouter is OpenRouterSTT

    embedding = lazyllm.OnlineEmbeddingModule(
        model='liquid/lfm-2.5-embedding-350m:free',
        source='openrouter',
        url='https://openrouter.ai/api/v1/',
        api_key='test-key',
    )
    assert embedding._embed_url == 'https://openrouter.ai/api/v1/embeddings'
    assert embedding._embed_model_name == 'liquid/lfm-2.5-embedding-350m:free'

    image = lazyllm.OnlineMultiModalModule(
        model='bytedance-seed/seedream-4.5',
        source='openrouter',
        url='https://openrouter.ai/api/v1/',
        api_key='test-key',
        type='text2image',
    )
    assert image._base_url == 'https://openrouter.ai/api/v1/'
    assert image._model_name == 'bytedance-seed/seedream-4.5'


def test_openrouter_image_uses_dedicated_images_api(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {'data': [{'b64_json': base64.b64encode(b'image-bytes').decode()}]}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(openrouter_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(openrouter_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(openrouter_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    image = OpenRouterText2Image(api_key='test-key')
    result = image._forward(input='a red panda')

    assert captured['url'] == 'https://openrouter.ai/api/v1/images'
    assert captured['json']['model'] == 'bytedance-seed/seedream-4.5'
    assert captured['json']['prompt'] == 'a red panda'
    assert result == [b'image-bytes']


def test_openrouter_video_uses_async_videos_api(monkeypatch):
    captured = {'get_urls': []}

    class Response:
        def __init__(self, *, data=None, content=b''):
            self._data = data
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_post(url, **kwargs):
        captured.update(post_url=url, post_kwargs=kwargs)
        return Response(data={
            'id': 'job-123',
            'polling_url': '/api/v1/videos/job-123',
            'status': 'pending',
        })

    def fake_get(url, **kwargs):
        captured['get_urls'].append(url)
        if url.endswith('/content'):
            return Response(content=b'video-bytes')
        return Response(data={
            'id': 'job-123',
            'polling_url': '/api/v1/videos/job-123',
            'status': 'completed',
        })

    monkeypatch.setattr(openrouter_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(openrouter_supplier.requests, 'get', fake_get)
    monkeypatch.setattr(openrouter_supplier.time, 'sleep', lambda _: None)
    monkeypatch.setattr(openrouter_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(openrouter_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    video = OpenRouterText2Video(api_key='test-key')
    result = video._forward(input='a bird flying', poll_interval=0)

    assert captured['post_url'] == 'https://openrouter.ai/api/v1/videos'
    assert captured['post_kwargs']['json']['model'] == 'bytedance/seedance-2.0-mini'
    assert captured['get_urls'] == [
        'https://openrouter.ai/api/v1/videos/job-123',
        'https://openrouter.ai/api/v1/videos/job-123/content',
    ]
    assert result == [b'video-bytes']


def test_openrouter_tts_uses_speech_api(monkeypatch):
    captured = {}

    class Response:
        content = b'audio-bytes'

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(openrouter_supplier.requests, 'post', fake_post)
    monkeypatch.setattr(openrouter_supplier, 'bytes_to_file', lambda values: values)
    monkeypatch.setattr(openrouter_supplier, 'encode_query_with_filepaths', lambda _, values: values)

    tts = OpenRouterTTS(api_key='test-key')
    result = tts._forward(input='hello')

    assert captured['url'] == 'https://openrouter.ai/api/v1/audio/speech'
    assert captured['json']['model'] == 'deepgram/flux-tts:free'
    assert captured['json']['input'] == 'hello'
    assert result == [b'audio-bytes']


def test_openrouter_stt_uses_transcriptions_api(monkeypatch, tmp_path):
    captured = {}
    audio_file = tmp_path / 'sample.wav'
    audio_file.write_bytes(b'wav-bytes')

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {'text': 'hello world'}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(openrouter_supplier.requests, 'post', fake_post)

    stt = OpenRouterSTT(api_key='test-key')
    result = stt._forward(input=str(audio_file), language='en')

    assert captured['url'] == 'https://openrouter.ai/api/v1/audio/transcriptions'
    assert captured['json']['model'] == 'openai/whisper-large-v3-turbo'
    assert captured['json']['input_audio'] == {
        'data': base64.b64encode(b'wav-bytes').decode('ascii'),
        'format': 'wav',
    }
    assert captured['json']['language'] == 'en'
    assert result == 'hello world'
