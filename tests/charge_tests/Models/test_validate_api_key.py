import pytest
import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.deepseek import DeepSeekChat
from lazyllm.module.llms.onlinemodule.supplier.doubao import DoubaoChat
from lazyllm.module.llms.onlinemodule.supplier.kimi import KimiChat
from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat
from lazyllm.module.llms.onlinemodule.supplier.glm import GLMChat
from lazyllm.module.llms.onlinemodule.supplier.qwen import QwenChat
from lazyllm.module.llms.onlinemodule.supplier.sensenova import SenseNovaChat
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowChat
from lazyllm.module.llms.onlinemodule.supplier.ppio import PPIOChat
from lazyllm.module.llms.onlinemodule.supplier.aiping import AipingChat

from ...utils import get_api_key


class TestValidateApiKey:
    '''Test online model API Key validation functionality'''

    @pytest.mark.xfail
    def test_deepseek_validate_valid_api_key(self):
        '''Test DeepSeek valid API Key validation'''
        module = DeepSeekChat(api_key=get_api_key('deepseek'))
        result = module._validate_api_key()
        assert result is True, 'DeepSeek valid API Key should pass validation'

    @pytest.mark.xfail
    def test_deepseek_validate_invalid_api_key(self):
        '''Test DeepSeek invalid API Key validation'''
        module = DeepSeekChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'DeepSeek invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_doubao_validate_valid_api_key(self):
        '''Test Doubao valid API Key validation'''
        module = DoubaoChat(api_key=get_api_key('doubao'), model=lazyllm.config['doubao_model_name'] or None)

        result = module._validate_api_key()
        assert result is True, 'Doubao valid API Key should pass validation'

    def test_doubao_validate_invalid_api_key(self):
        '''Test Doubao invalid API Key validation'''
        module = DoubaoChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Doubao invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_kimi_validate_valid_api_key(self):
        '''Test Kimi valid API Key validation'''
        module = KimiChat(api_key=get_api_key('kimi'))
        result = module._validate_api_key()
        assert result is True, 'Kimi valid API Key should pass validation'

    @pytest.mark.xfail
    def test_kimi_validate_invalid_api_key(self):
        '''Test Kimi invalid API Key validation'''
        module = KimiChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Kimi invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_openai_validate_valid_api_key(self):
        '''Test OpenAI valid API Key validation'''
        module = OpenAIChat(api_key=get_api_key('openai'))
        result = module._validate_api_key()
        assert result is True, 'OpenAI valid API Key should pass validation'

    @pytest.mark.xfail
    def test_openai_validate_invalid_api_key(self):
        '''Test OpenAI invalid API Key validation'''
        module = OpenAIChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'OpenAI invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_glm_validate_valid_api_key(self):
        '''Test GLM valid API Key validation'''
        module = GLMChat(api_key=get_api_key('glm'))
        result = module._validate_api_key()
        assert result is True, 'GLM valid API Key should pass validation'

    @pytest.mark.xfail
    def test_glm_validate_invalid_api_key(self):
        '''Test GLM invalid API Key validation'''
        module = GLMChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'GLM invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_qwen_validate_valid_api_key(self):
        '''Test Qwen valid API Key validation'''
        module = QwenChat(api_key=get_api_key('qwen'))
        result = module._validate_api_key()
        assert result is True, 'Qwen valid API Key should pass validation'

    def test_qwen_validate_invalid_api_key(self):
        '''Test Qwen invalid API Key validation'''
        module = QwenChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Qwen invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_sensenova_validate_valid_api_key(self):
        '''Test SenseNova valid API Key validation'''
        module = SenseNovaChat(
            api_key=get_api_key('sensenova'),
            secret_key=lazyllm.config['sensenova_secret_key']
        )
        result = module._validate_api_key()
        assert result is True, 'SenseNova valid API Key should pass validation'

    def test_sensenova_validate_invalid_api_key(self):
        '''Test SenseNova invalid API Key validation'''
        module = SenseNovaChat(
            api_key='invalid_api_key_12345',
            secret_key='invalid_secret_key_12345'
        )
        result = module._validate_api_key()
        assert result is False, 'SenseNova invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_siliconflow_validate_valid_api_key(self):
        '''Test SiliconFlow valid API Key validation'''
        module = SiliconFlowChat(api_key=get_api_key('siliconflow'))
        result = module._validate_api_key()
        assert result is True, 'SiliconFlow valid API Key should pass validation'

    @pytest.mark.xfail
    def test_siliconflow_validate_invalid_api_key(self):
        '''Test SiliconFlow invalid API Key validation'''
        module = SiliconFlowChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'SiliconFlow invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_ppio_validate_valid_api_key(self):
        '''Test PPIO valid API Key validation'''
        module = PPIOChat(api_key=get_api_key('ppio'))
        result = module._validate_api_key()
        assert result is True, 'PPIO valid API Key should pass validation'

    @pytest.mark.xfail
    def test_ppio_validate_invalid_api_key(self):
        '''Test PPIO invalid API Key validation'''
        module = PPIOChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'PPIO invalid API Key should fail validation'

    @pytest.mark.xfail
    def test_aiping_validate_valid_api_key(self):
        '''Test Aiping valid API Key validation'''
        module = AipingChat(api_key=get_api_key('aiping'))
        result = module._validate_api_key()
        assert result is True, 'Aiping valid API Key should pass validation'

    @pytest.mark.xfail
    def test_aiping_validate_invalid_api_key(self):
        '''Test Aiping invalid API Key validation'''
        module = AipingChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Aiping invalid API Key should fail validation'
