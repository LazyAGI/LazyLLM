import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'


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
        api_key = get_api_key(source)
        t2i = lazyllm.OnlineMultiModalModule(source=source, type=type, api_key=api_key, **kwargs)
        result = t2i(self.test_image_prompt)
        self._check_file_result(result, 'image')
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_text2image(self):
        self.common_text2image(source='qwen', type='sd')

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail
    def test_doubao_text2image(self):
        self.common_text2image(source='doubao', type='text2image')

    @pytest.mark.ignore_cache_on_change(get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_text2image(self):
        self.common_text2image(source='siliconflow', type='text2image', model='Qwen/Qwen-Image')

    @pytest.mark.ignore_cache_on_change(get_path('glm'))
    @pytest.mark.xfail
    def test_glm_text2image(self):
        self.common_text2image(source='glm', type='text2image')

    @pytest.mark.ignore_cache_on_change(get_path('minimax'))
    @pytest.mark.xfail
    def test_minimax_text2image(self):
        self.common_text2image(source='minimax', function='text2image')

    @pytest.mark.ignore_cache_on_change(get_path('aiping'))
    @pytest.mark.xfail
    def test_aiping_text2image(self):
        self.common_text2image(source='aiping', type='text2image')
