import pytest

import lazyllm


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineEmbeddingModuleBase.py'
SILICONFLOW_PATH = 'lazyllm/module/llms/onlinemodule/supplier/siliconflow.py'
AIPING_PATH = 'lazyllm/module/llms/onlinemodule/supplier/aiping.py'
QWEN_PATH = 'lazyllm/module/llms/onlinemodule/supplier/qwen.py'
GLM_PATH = 'lazyllm/module/llms/onlinemodule/supplier/glm.py'
OPENAI_PATH = 'lazyllm/module/llms/onlinemodule/supplier/openai.py'

pytestmark = pytest.mark.model_connectivity_test

RERANK_CASES = [
    pytest.param('siliconflow', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, SILICONFLOW_PATH), id='siliconflow'),
    pytest.param('aiping', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, AIPING_PATH), id='aiping'),
    pytest.param('qwen', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, QWEN_PATH), id='qwen'),
    pytest.param('glm', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, GLM_PATH), id='glm'),
    pytest.param('openai', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, OPENAI_PATH), id='openai'),
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
