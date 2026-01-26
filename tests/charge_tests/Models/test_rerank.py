import pytest

import lazyllm

from tests.utils import get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineEmbeddingModuleBase.py'

pytestmark = pytest.mark.model_connectivity_test

RERANK_CASES = [
    pytest.param('siliconflow', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('siliconflow')), id='siliconflow'),
    pytest.param('aiping', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('aiping')), id='aiping'),
    pytest.param('qwen', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen')), id='qwen'),
    pytest.param('glm', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm')), id='glm'),
    pytest.param('openai', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('openai')), id='openai'),
]


class TestRerank:
    def common_rerank(self, source, **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
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

    @pytest.mark.parametrize('source, init_kwargs', RERANK_CASES)
    def test_rerank(self, source, init_kwargs):
        self.common_rerank(source=source, **init_kwargs)
