import pytest
import os
import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.deepseek import DeepSeekModule
from lazyllm.module.llms.onlinemodule.supplier.doubao import DoubaoModule
from lazyllm.module.llms.onlinemodule.supplier.kimi import KimiModule


class TestValidateApiKey:
    """测试在线模型 API Key 验证功能"""

    def setup_method(self):
        """初始化测试环境，从环境变量获取 API Keys"""
        self.api_keys = {
            'deepseek': os.getenv('LAZYLLM_DEEPSEEK_API_KEY'),
            'doubao': os.getenv('LAZYLLM_DOUBAO_API_KEY'),
            'kimi': os.getenv('LAZYLLM_KIMI_API_KEY'),
        }
        # 获取 doubao 模型名称（如果环境变量中有的话）
        self.doubao_model_name = os.getenv('LAZYLLM_DOUBAO_MODEL_NAME')

    def test_deepseek_validate_valid_api_key(self):
        """测试 DeepSeek 有效 API Key 验证"""
        if not self.api_keys['deepseek']:
            pytest.skip("未设置 LAZYLLM_DEEPSEEK_API_KEY 环境变量")

        module = DeepSeekModule(api_key=self.api_keys['deepseek'])
        result = module._validate_api_key()
        assert result is True, "DeepSeek 有效的 API Key 应该验证通过"

    def test_deepseek_validate_invalid_api_key(self):
        """测试 DeepSeek 无效 API Key 验证"""
        module = DeepSeekModule(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, "DeepSeek 无效的 API Key 应该验证失败"

    def test_doubao_validate_valid_api_key(self):
        """测试 Doubao 有效 API Key 验证"""
        if not self.api_keys['doubao']:
            pytest.skip("未设置 LAZYLLM_DOUBAO_API_KEY 环境变量")

        # 如果环境变量中设置了模型名称，使用该名称；否则使用默认值
        if self.doubao_model_name:
            module = DoubaoModule(
                api_key=self.api_keys['doubao'],
                model=self.doubao_model_name
            )
        else:
            module = DoubaoModule(api_key=self.api_keys['doubao'])

        result = module._validate_api_key()
        assert result is True, "Doubao 有效的 API Key 应该验证通过"

    def test_doubao_validate_invalid_api_key(self):
        """测试 Doubao 无效 API Key 验证"""
        module = DoubaoModule(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, "Doubao 无效的 API Key 应该验证失败"

    def test_kimi_validate_valid_api_key(self):
        """测试 Kimi 有效 API Key 验证"""
        if not self.api_keys['kimi']:
            pytest.skip("未设置 LAZYLLM_KIMI_API_KEY 环境变量")

        module = KimiModule(api_key=self.api_keys['kimi'])
        result = module._validate_api_key()
        assert result is True, "Kimi 有效的 API Key 应该验证通过"

    def test_kimi_validate_invalid_api_key(self):
        """测试 Kimi 无效 API Key 验证"""
        module = KimiModule(api_key='invalid_api_key_12345')
        result = module._validate_api_key()
        assert result is False, "Kimi 无效的 API Key 应该验证失败"

