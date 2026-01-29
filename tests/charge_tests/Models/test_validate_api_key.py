import pytest
import os
from lazyllm.module.llms.onlinemodule.supplier.deepseek import DeepSeekChat
from lazyllm.module.llms.onlinemodule.supplier.doubao import DoubaoChat
from lazyllm.module.llms.onlinemodule.supplier.kimi import KimiChat
from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat
from lazyllm.module.llms.onlinemodule.supplier.glm import GLMChat
from lazyllm.module.llms.onlinemodule.supplier.qwen import QwenChat
from lazyllm.module.llms.onlinemodule.supplier.sensenova import SenseNovaChat
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowChat
from lazyllm.module.llms.onlinemodule.supplier.ppio import PPIOChat
from lazyllm.module.llms.onlinemodule.supplier.intel import IntelChat
from lazyllm.module.llms.onlinemodule.supplier.moark import MoarkChat
from lazyllm.module.llms.onlinemodule.supplier.aiping import AipingChat


class TestValidateApiKey:
    '''Test online model API Key validation functionality'''

    def setup_method(self):
        '''Initialize test environment, get API Keys from environment variables'''
        self.api_keys = {
            'deepseek': os.getenv('LAZYLLM_DEEPSEEK_API_KEY'),
            'doubao': os.getenv('LAZYLLM_DOUBAO_API_KEY'),
            'kimi': os.getenv('LAZYLLM_KIMI_API_KEY'),
            'openai': os.getenv('LAZYLLM_OPENAI_API_KEY'),
            'glm': os.getenv('LAZYLLM_GLM_API_KEY'),
            'qwen': os.getenv('LAZYLLM_QWEN_API_KEY'),
            'sensenova': os.getenv('LAZYLLM_SENSENOVA_API_KEY'),
            'siliconflow': os.getenv('LAZYLLM_SILICONFLOW_API_KEY'),
            'ppio': os.getenv('LAZYLLM_PPIO_API_KEY'),
            'intel': os.getenv('LAZYLLM_INTEL_API_KEY'),
            'moark': os.getenv('LAZYLLM_MOARK_API_KEY'),
            'aiping': os.getenv('LAZYLLM_AIPING_API_KEY'),
        }
        # Get sensenova secret key
        self.sensenova_secret_key = os.getenv('LAZYLLM_SENSENOVA_SECRET_KEY')
        # Get doubao model name (if set in environment variables)
        self.doubao_model_name = os.getenv('LAZYLLM_DOUBAO_MODEL_NAME')

    def test_deepseek_validate_valid_api_key(self):
        '''Test DeepSeek valid API Key validation'''
        if not self.api_keys['deepseek']:
            pytest.skip('LAZYLLM_DEEPSEEK_API_KEY environment variable is not set')

        module = DeepSeekChat(api_key=self.api_keys['deepseek'])
        result = module._validate_api_key()
        assert result is True, 'DeepSeek valid API Key should pass validation'

    def test_deepseek_validate_invalid_api_key(self):
        '''Test DeepSeek invalid API Key validation'''
        module = DeepSeekChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'DeepSeek invalid API Key should fail validation'

    def test_doubao_validate_valid_api_key(self):
        '''Test Doubao valid API Key validation'''
        if not self.api_keys['doubao']:
            pytest.skip('LAZYLLM_DOUBAO_API_KEY environment variable is not set')

        # If model name is set in environment variables, use it; otherwise use default
        if self.doubao_model_name:
            module = DoubaoChat(
                api_key=self.api_keys['doubao'],
                model=self.doubao_model_name
            )
        else:
            module = DoubaoChat(api_key=self.api_keys['doubao'])

        result = module._validate_api_key()
        assert result is True, 'Doubao valid API Key should pass validation'

    def test_doubao_validate_invalid_api_key(self):
        '''Test Doubao invalid API Key validation'''
        module = DoubaoChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Doubao invalid API Key should fail validation'

    def test_kimi_validate_valid_api_key(self):
        '''Test Kimi valid API Key validation'''
        if not self.api_keys['kimi']:
            pytest.skip('LAZYLLM_KIMI_API_KEY environment variable is not set')

        module = KimiChat(api_key=self.api_keys['kimi'])
        result = module._validate_api_key()
        assert result is True, 'Kimi valid API Key should pass validation'

    def test_kimi_validate_invalid_api_key(self):
        '''Test Kimi invalid API Key validation'''
        module = KimiChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Kimi invalid API Key should fail validation'

    def test_openai_validate_valid_api_key(self):
        '''Test OpenAI valid API Key validation'''
        if not self.api_keys['openai']:
            pytest.skip('LAZYLLM_OPENAI_API_KEY environment variable is not set')

        module = OpenAIChat(api_key=self.api_keys['openai'])
        result = module._validate_api_key()
        assert result is True, 'OpenAI valid API Key should pass validation'

    def test_openai_validate_invalid_api_key(self):
        '''Test OpenAI invalid API Key validation'''
        module = OpenAIChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'OpenAI invalid API Key should fail validation'

    def test_glm_validate_valid_api_key(self):
        '''Test GLM valid API Key validation'''
        if not self.api_keys['glm']:
            pytest.skip('LAZYLLM_GLM_API_KEY environment variable is not set')

        module = GLMChat(api_key=self.api_keys['glm'])
        result = module._validate_api_key()
        assert result is True, 'GLM valid API Key should pass validation'

    def test_glm_validate_invalid_api_key(self):
        '''Test GLM invalid API Key validation'''
        module = GLMChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'GLM invalid API Key should fail validation'

    def test_qwen_validate_valid_api_key(self):
        '''Test Qwen valid API Key validation'''
        if not self.api_keys['qwen']:
            pytest.skip('LAZYLLM_QWEN_API_KEY environment variable is not set')

        module = QwenChat(api_key=self.api_keys['qwen'])
        result = module._validate_api_key()
        assert result is True, 'Qwen valid API Key should pass validation'

    def test_qwen_validate_invalid_api_key(self):
        '''Test Qwen invalid API Key validation'''
        module = QwenChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Qwen invalid API Key should fail validation'

    def test_sensenova_validate_valid_api_key(self):
        '''Test SenseNova valid API Key validation'''
        if not self.api_keys['sensenova'] or not self.sensenova_secret_key:
            pytest.skip('LAZYLLM_SENSENOVA_API_KEY or LAZYLLM_SENSENOVA_SECRET_KEY environment variable is not set')

        module = SenseNovaChat(
            api_key=self.api_keys['sensenova'],
            secret_key=self.sensenova_secret_key
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

    def test_siliconflow_validate_valid_api_key(self):
        '''Test SiliconFlow valid API Key validation'''
        if not self.api_keys['siliconflow']:
            pytest.skip('LAZYLLM_SILICONFLOW_API_KEY environment variable is not set')

        module = SiliconFlowChat(api_key=self.api_keys['siliconflow'])
        result = module._validate_api_key()
        assert result is True, 'SiliconFlow valid API Key should pass validation'

    def test_siliconflow_validate_invalid_api_key(self):
        '''Test SiliconFlow invalid API Key validation'''
        module = SiliconFlowChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'SiliconFlow invalid API Key should fail validation'

    def test_ppio_validate_valid_api_key(self):
        '''Test PPIO valid API Key validation'''
        if not self.api_keys['ppio']:
            pytest.skip('LAZYLLM_PPIO_API_KEY environment variable is not set')

        module = PPIOChat(api_key=self.api_keys['ppio'])
        result = module._validate_api_key()
        assert result is True, 'PPIO valid API Key should pass validation'

    def test_ppio_validate_invalid_api_key(self):
        '''Test PPIO invalid API Key validation'''
        module = PPIOChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'PPIO invalid API Key should fail validation'

    def test_intel_validate_valid_api_key(self):
        '''Test Intel valid API Key validation'''
        if not self.api_keys['intel']:
            pytest.skip('LAZYLLM_INTEL_API_KEY environment variable is not set')

        module = IntelModule(api_key=self.api_keys['intel'])
        result = module._validate_api_key()
        assert result is True, 'Intel valid API Key should pass validation'
    def test_intel_validate_invalid_api_key(self):
        '''Test Intel invalid API Key validation'''
        module = IntelModule(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Intel invalid API Key should fail validation'

    def test_moark_validate_valid_api_key(self):
        '''Test Moark valid API Key validation'''
        if not self.api_keys['moark']:
            pytest.skip('LAZYLLM_MOARK_API_KEY environment variable is not set')

        module = MoarkModule(api_key=self.api_keys['moark'])
        result = module._validate_api_key()
        assert result is True, 'Moark valid API Key should pass validation'
    def test_moark_validate_invalid_api_key(self):
        '''Test Moark invalid API Key validation'''
        module = MoarkModule(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Moark invalid API Key should fail validation'

    def test_aiping_validate_valid_api_key(self):
        '''Test Aiping valid API Key validation'''
        if not self.api_keys['aiping']:
            pytest.skip('LAZYLLM_AIPING_API_KEY environment variable is not set')

        module = AipingChat(api_key=self.api_keys['aiping'])
        result = module._validate_api_key()
        assert result is True, 'Aiping valid API Key should pass validation'

    def test_aiping_validate_invalid_api_key(self):
        '''Test Aiping invalid API Key validation'''
        module = AipingChat(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, 'Aiping invalid API Key should fail validation'
