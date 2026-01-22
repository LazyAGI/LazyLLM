import pytest

import lazyllm


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py'
QWEN_PATH = 'lazyllm/module/llms/onlinemodule/supplier/qwen.py'
DOUBAO_PATH = 'lazyllm/module/llms/onlinemodule/supplier/doubao.py'
GLM_PATH = 'lazyllm/module/llms/onlinemodule/supplier/glm.py'
SILICONFLOW_PATH = 'lazyllm/module/llms/onlinemodule/supplier/siliconflow.py'
MINIMAX_PATH = 'lazyllm/module/llms/onlinemodule/supplier/minimax.py'
SENSENOVA_PATH = 'lazyllm/module/llms/onlinemodule/supplier/sensenova.py'
OPENAI_PATH = 'lazyllm/module/llms/onlinemodule/supplier/openai.py'
DEEPSEEK_PATH = 'lazyllm/module/llms/onlinemodule/supplier/deepseek.py'
KIMI_PATH = 'lazyllm/module/llms/onlinemodule/supplier/kimi.py'
AIPING_PATH = 'lazyllm/module/llms/onlinemodule/supplier/aiping.py'
PPIO_PATH = 'lazyllm/module/llms/onlinemodule/supplier/ppio.py'

pytestmark = pytest.mark.model_connectivity_test

CHAT_CASES = [
    pytest.param('qwen', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, QWEN_PATH), id='qwen'),
    pytest.param('doubao', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, DOUBAO_PATH), id='doubao'),
    pytest.param('glm', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, GLM_PATH), id='glm'),
    pytest.param('siliconflow', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, SILICONFLOW_PATH), id='siliconflow'),
    pytest.param('minimax', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, MINIMAX_PATH), id='minimax'),
    pytest.param('sensenova', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, SENSENOVA_PATH), id='sensenova'),
    pytest.param('openai', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, OPENAI_PATH), id='openai'),
    pytest.param('deepseek', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, DEEPSEEK_PATH), id='deepseek'),
    pytest.param('kimi', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, KIMI_PATH), id='kimi'),
    pytest.param('aiping', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, AIPING_PATH), id='aiping'),
    pytest.param('ppio', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, PPIO_PATH), id='ppio'),
]


class TestChat:
    def common_chat(self, source, query='你好，请介绍一下你自己', **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
        chat = lazyllm.OnlineModule(source=source, type='llm', api_key=api_key, **kwargs)
        result = chat(query)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        return result

    @pytest.mark.parametrize('source, init_kwargs', CHAT_CASES)
    def test_chat(self, source, init_kwargs):
        self.common_chat(source=source, **init_kwargs)
