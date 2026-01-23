import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'
QWEN_PATH = 'lazyllm/module/llms/onlinemodule/supplier/qwen.py'
DOUBAO_PATH = 'lazyllm/module/llms/onlinemodule/supplier/doubao.py'
GLM_PATH = 'lazyllm/module/llms/onlinemodule/supplier/glm.py'
SILICONFLOW_PATH = 'lazyllm/module/llms/onlinemodule/supplier/siliconflow.py'
MINIMAX_PATH = 'lazyllm/module/llms/onlinemodule/supplier/minimax.py'
AIPING_PATH = 'lazyllm/module/llms/onlinemodule/supplier/aiping.py'

pytestmark = pytest.mark.model_connectivity_test

TEXT2IMAGE_CASES = [
    pytest.param('qwen', {'type': 'sd'}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, QWEN_PATH), id='qwen'),
    pytest.param('doubao', {'type': 'text2image'}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, DOUBAO_PATH), id='doubao'),
    pytest.param('siliconflow', {'type': 'text2image', 'model': 'Qwen/Qwen-Image'},
                 marks=pytest.mark.ignore_cache_on_change(BASE_PATH, SILICONFLOW_PATH), id='siliconflow'),
    pytest.param('glm', {'type': 'text2image'}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, GLM_PATH), id='glm'),
    pytest.param('minimax', {'function': 'text2image'}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, MINIMAX_PATH), id='minimax'),
    pytest.param('aiping', {'type': 'text2image'}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, AIPING_PATH), id='aiping'),
]


class TestText2Image:
    test_image_prompt = '画一只动漫风格的懒懒猫'

    @staticmethod
    def _check_file_result(result, format_type='image'):
        assert result is not None
        assert isinstance(result, str)
        assert result.startswith('<lazyllm-query>')

        decoded = decode_query_with_filepaths(result)
        assert 'files' in decoded
        assert len(decoded['files']) > 0

        file = decoded['files'][0]
        assert os.path.exists(file)
        suffix = ('.png', '.jpg', '.jpeg') if format_type == 'image' else ('.wav', '.mp3', '.flac')
        assert file.endswith(suffix)

    def common_text2image(self, source, type='text2image', **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
        t2i = lazyllm.OnlineMultiModalModule(source=source, type=type, api_key=api_key, **kwargs)
        result = t2i(self.test_image_prompt)
        self._check_file_result(result, 'image')
        return result

    @pytest.mark.parametrize('source, init_kwargs', TEXT2IMAGE_CASES)
    def test_text2image(self, source, init_kwargs):
        self.common_text2image(source=source, **init_kwargs)
