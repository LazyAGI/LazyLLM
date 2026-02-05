import os

import pytest

import lazyllm
from lazyllm.module.module import ModuleExecutionError
from lazyllm.components.formatter import decode_query_with_filepaths

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'


class TestTTS:
    test_text = '你好，这是一个测试。'

    @staticmethod
    def _check_file_result(result):
        assert result is not None
        assert isinstance(result, str)
        assert result.startswith('<lazyllm-query>')

        decoded = decode_query_with_filepaths(result)
        assert 'files' in decoded
        assert len(decoded['files']) > 0

        file = decoded['files'][0]
        assert os.path.exists(file)
        assert file.endswith(('.wav', '.mp3', '.flac'))

    def common_tts(self, source, call_kwargs=None, **kwargs):
        api_key = get_api_key(source)
        tts = lazyllm.OnlineMultiModalModule(source=source, function='tts', api_key=api_key, **kwargs)
        result = tts(self.test_text, **(call_kwargs or {}))
        self._check_file_result(result)
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_tts(self):
        self.common_tts(source='qwen', model='qwen-tts')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('minimax'))
    @pytest.mark.xfail
    def test_minimax_tts(self):
        self.common_tts(source='minimax')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_tts(self):
        self.common_tts(source='siliconflow', call_kwargs={'voice': 'fnlp/MOSS-TTSD-v0.5:anna'})

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_tts_cosyvoice_multi_user_raises(self):
        api_key = get_api_key('qwen')
        with pytest.raises(ModuleExecutionError, match="cosyvoice-v1 does not support multi user, don't set api_key"):
            tts = lazyllm.OnlineMultiModalModule(source='qwen', function='tts', model='cosyvoice-v1', api_key=api_key)
            tts(self.test_text)
