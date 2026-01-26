import pytest

import lazyllm

from tests.utils import get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineEmbeddingModuleBase.py'

pytestmark = pytest.mark.model_connectivity_test

EMBEDDING_CASES = [
    pytest.param('qwen', {'embed_model_name': 'text-embedding-v3'},
                 marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen')), id='qwen'),
    pytest.param('doubao', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('doubao')), id='doubao'),
    pytest.param('glm', {}, marks=[
        pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm')), pytest.mark.xfail], id='glm'),
    pytest.param('siliconflow', {}, marks=[pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('siliconflow')), pytest.mark.xfail], id='siliconflow'),
    pytest.param('openai', {}, marks=[
        pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('openai')), pytest.mark.xfail], id='openai'),
    pytest.param('aiping', {}, marks=[
        pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('aiping')), pytest.mark.xfail], id='aiping'),
    pytest.param('sensenova', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('sensenova')), id='sensenova'),
]

MULTIMODAL_EMBEDDING_CASES = [
    pytest.param(
        'doubao',
        {},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('doubao')),
        id='doubao_multimodal',
    ),
]


class TestEmbedding:
    def common_embedding(self, source, **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
        embed_model = lazyllm.OnlineEmbeddingModule(source=source, api_key=api_key, **kwargs)
        vec1 = embed_model('床前明月光')
        vec2 = embed_model(['床前明月光', '疑是地上霜'])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)
        return vec2

    def common_multimodal_embedding(self, source, **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
        embed_model = lazyllm.OnlineEmbeddingModule(source=source, api_key=api_key, **kwargs)
        vec = embed_model([{'text': '床前明月光'}])
        assert vec is not None
        assert isinstance(vec, list)
        assert len(vec) > 0
        vec_f = [float(x) for x in vec]
        assert all(x == x for x in vec_f)
        return vec

    @pytest.mark.parametrize('source, init_kwargs', EMBEDDING_CASES)
    def test_embedding(self, source, init_kwargs):
        self.common_embedding(source=source, **init_kwargs)

    @pytest.mark.parametrize('source, init_kwargs', MULTIMODAL_EMBEDDING_CASES)
    def test_multimodal_embedding(self, source, init_kwargs):
        if source == 'doubao':
            key = 'doubao_multimodal_embed_model_name'
            model_name = lazyllm.config[key] if key in lazyllm.config.get_all_configs() else ''
            if not model_name:
                pytest.skip(f'{key} is not configured, skipping doubao multimodal embedding connectivity test.')
            if not model_name.startswith('doubao-embedding-vision'):
                pytest.skip(f'{key} must start with "doubao-embedding-vision" to select multimodal embed module.')
            init_kwargs = dict(init_kwargs)
            init_kwargs['embed_model_name'] = model_name
        self.common_multimodal_embedding(source=source, **init_kwargs)
