import pytest

import lazyllm

from tests.utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineEmbeddingModuleBase.py'


class TestRerank:
    def common_rerank(self, source, **kwargs):
        api_key = get_api_key(source)
        rerank_model = lazyllm.OnlineEmbeddingModule(source=source, api_key=api_key, type='rerank', **kwargs)
        result = rerank_model(
            '床前明月光',
            documents=['床前明月光', '疑是地上霜', '举头望明月', '低头思故乡'],
            top_n=4,
        )
        indices = [item[0] for item in result]
        scores = [float(item[1]) for item in result]
        assert len(result) == 4
        assert set(indices) == {0, 1, 2, 3}
        for _, score in result:
            score_f = float(score)
            assert score_f == score_f
        assert scores[0] == max(scores)
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_rerank(self):
        self.common_rerank(source='qwen')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_rerank(self):
        self.common_rerank(source='siliconflow')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('aiping'))
    @pytest.mark.xfail
    def test_aiping_rerank(self):
        self.common_rerank(source='aiping')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm'))
    @pytest.mark.xfail
    def test_glm_rerank(self):
        self.common_rerank(source='glm')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('openai'))
    @pytest.mark.xfail
    def test_openai_rerank(self):
        self.common_rerank(source='openai')
