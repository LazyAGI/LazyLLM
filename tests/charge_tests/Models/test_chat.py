import pytest

import lazyllm

from tests.utils import get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py'

pytestmark = pytest.mark.model_connectivity_test

CHAT_CASES = [
    pytest.param('qwen', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen')), id='qwen'),
    pytest.param('doubao', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('doubao')), id='doubao'),
    pytest.param('glm', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm')), id='glm'),
    pytest.param('siliconflow', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('siliconflow')), id='siliconflow'),
    pytest.param('minimax', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('minimax')), id='minimax'),
    pytest.param('sensenova', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('sensenova')), id='sensenova'),
    pytest.param('openai', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('openai')), id='openai'),
    pytest.param('deepseek', {}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('deepseek')), id='deepseek'),
    pytest.param('kimi', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('kimi')), id='kimi'),
    pytest.param('aiping', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('aiping')), id='aiping'),
    pytest.param('ppio', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('ppio')), id='ppio'),
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
