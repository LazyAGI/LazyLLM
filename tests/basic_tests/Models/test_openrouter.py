import base64

import lazyllm

from lazyllm.module.llms.onlinemodule.supplier import openrouter as openrouter_supplier

from lazyllm.module.llms.onlinemodule.supplier.openrouter import (
    OpenRouterChat,
    OpenRouterEmbed,
    OpenRouterText2Image,
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


def test_openrouter_embedding_and_image_registration():
    assert lazyllm.online.embed.openrouter is OpenRouterEmbed
    assert lazyllm.online.text2image.openrouter is OpenRouterText2Image

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
