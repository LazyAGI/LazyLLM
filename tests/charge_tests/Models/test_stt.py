import os

import pytest

import lazyllm

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'


class TestSTT:
    QWEN_TEST_AUDIO_URL = (
        'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav'
    )

    @staticmethod
    def _test_audio_file():
        data_path = lazyllm.config['data_path']
        return os.path.join(data_path, 'ci_data/asr_test.wav')

    def common_stt(self, source, **kwargs):
        api_key = get_api_key(source)
        stt = lazyllm.OnlineMultiModalModule(source=source, function='stt', api_key=api_key, **kwargs)
        test_input = [self.QWEN_TEST_AUDIO_URL] if source == 'qwen' else self._test_audio_file()
        result = stt(lazyllm_files=test_input)
        assert isinstance(result, str)
        assert len(result) > 0
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_stt(self):
        self.common_stt(source='qwen')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm'))
    @pytest.mark.xfail
    def test_glm_stt(self):
        self.common_stt(source='glm')
