import pytest

import lazyllm

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py'


class TestChat:
    def common_chat(self, source, query='你好，请介绍一下你自己', **kwargs):
        api_key = get_api_key(source)
        chat = lazyllm.OnlineModule(source=source, type='llm', api_key=api_key, **kwargs)
        result = chat(query)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_chat(self):
        self.common_chat(source='qwen')

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail
    def test_doubao_chat(self):
        self.common_chat(source='doubao')

    @pytest.mark.ignore_cache_on_change(get_path('glm'))
    @pytest.mark.xfail
    def test_glm_chat(self):
        self.common_chat(source='glm')

    @pytest.mark.ignore_cache_on_change(get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_chat(self):
        self.common_chat(source='siliconflow')

    @pytest.mark.ignore_cache_on_change(get_path('minimax'))
    @pytest.mark.xfail
    def test_minimax_chat(self):
        self.common_chat(source='minimax')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('sensenova'))
    def test_sensenova_chat(self):
        self.common_chat(source='sensenova')

    @pytest.mark.ignore_cache_on_change(get_path('openai'))
    @pytest.mark.xfail
    def test_openai_chat(self):
        self.common_chat(source='openai')

    @pytest.mark.ignore_cache_on_change(get_path('deepseek'))
    @pytest.mark.xfail
    def test_deepseek_chat(self):
        self.common_chat(source='deepseek')

    @pytest.mark.ignore_cache_on_change(get_path('kimi'))
    @pytest.mark.xfail
    def test_kimi_chat(self):
        self.common_chat(source='kimi')

    @pytest.mark.ignore_cache_on_change(get_path('aiping'))
    @pytest.mark.xfail
    def test_aiping_chat(self):
        self.common_chat(source='aiping')

    @pytest.mark.ignore_cache_on_change(get_path('ppio'))
    @pytest.mark.xfail
    def test_ppio_chat(self):
        self.common_chat(source='ppio')
