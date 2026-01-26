import os

import pytest

import lazyllm
from lazyllm import config
from lazyllm.components.formatter import decode_query_with_filepaths

from tests.utils import get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'

pytestmark = pytest.mark.model_connectivity_test

TTS_CASES = [
    pytest.param('qwen', {'model': 'qwen-tts'}, marks=pytest.mark.ignore_cache_on_change(
        BASE_PATH, get_path('qwen')), id='qwen'),
    pytest.param('minimax', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('minimax')), id='minimax'),
    pytest.param(
        'siliconflow',
        {'call_kwargs': {'voice': 'fnlp/MOSS-TTSD-v0.5:anna'}},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('siliconflow')),
        id='siliconflow',
    ),
]


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
        api_key = lazyllm.config[f'{source}_api_key']
        tts = lazyllm.OnlineMultiModalModule(source=source, function='tts', api_key=api_key, **kwargs)
        result = tts(self.test_text, **(call_kwargs or {}))
        self._check_file_result(result)
        return result

    @pytest.mark.parametrize('source, init_kwargs', TTS_CASES)
    def test_tts(self, source, init_kwargs):
        self.common_tts(source=source, **init_kwargs)

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_tts_cosyvoice_multi_user_raises(self):
        if config['cache_online_module']:
            return
        api_key = lazyllm.config['qwen_api_key']
        with pytest.raises(RuntimeError, match="cosyvoice-v1 does not support multi user, don't set api_key"):
            tts = lazyllm.OnlineMultiModalModule(source='qwen', function='tts', model='cosyvoice-v1', api_key=api_key)
            tts(self.test_text)
