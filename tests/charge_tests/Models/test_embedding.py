import pytest

import lazyllm

from tests.utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineEmbeddingModuleBase.py'


class TestEmbedding:
    def common_embedding(self, source, **kwargs):
        api_key = get_api_key(source)
        embed_model = lazyllm.OnlineEmbeddingModule(source=source, api_key=api_key, **kwargs)
        vec1 = embed_model('床前明月光')
        vec2 = embed_model(['床前明月光', '疑是地上霜'])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)
        return vec2

    def common_multimodal_embedding(self, source, **kwargs):
        api_key = get_api_key(source)
        embed_model = lazyllm.OnlineEmbeddingModule(source=source, api_key=api_key, **kwargs)
        vec = embed_model([{'text': '床前明月光'}])
        assert vec is not None
        assert isinstance(vec, list)
        assert len(vec) > 0
        vec_f = [float(x) for x in vec]
        assert all(x == x for x in vec_f)
        return vec

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_embedding(self):
        self.common_embedding(source='qwen', embed_model_name='text-embedding-v3')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('doubao'))
    def test_doubao_embedding(self):
        self.common_embedding(source='doubao')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm'))
    @pytest.mark.xfail
    def test_glm_embedding(self):
        self.common_embedding(source='glm')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_embedding(self):
        self.common_embedding(source='siliconflow')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('openai'))
    @pytest.mark.xfail
    def test_openai_embedding(self):
        self.common_embedding(source='openai')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('aiping'))
    @pytest.mark.xfail
    def test_aiping_embedding(self):
        self.common_embedding(source='aiping')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('sensenova'))
    def test_sensenova_embedding(self):
        self.common_embedding(source='sensenova')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('doubao'))
    @pytest.mark.xfail
    def test_doubao_multimodal_embedding(self):
        model_name = (lazyllm.config['doubao_multimodal_embed_model_name']
                      or 'doubao-embedding-vision-241215')
        self.common_multimodal_embedding(source='doubao', embed_model_name=model_name)
