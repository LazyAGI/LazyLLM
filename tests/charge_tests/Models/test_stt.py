import os

import pytest

import lazyllm

from tests.utils import get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'

pytestmark = pytest.mark.model_connectivity_test

STT_CASES = [
    pytest.param('glm', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('glm')), id='glm'),
    pytest.param('qwen', {}, marks=pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen')), id='qwen'),
]


class TestSTT:
    QWEN_TEST_AUDIO_URL = (
        'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav'
    )

    @staticmethod
    def _test_audio_file():
        data_path = lazyllm.config['data_path']
        return os.path.join(data_path, 'ci_data/asr_test.wav')

    def common_stt(self, source, **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
        stt = lazyllm.OnlineMultiModalModule(source=source, function='stt', api_key=api_key, **kwargs)
        test_input = [self.QWEN_TEST_AUDIO_URL] if source == 'qwen' else self._test_audio_file()
        result = stt(lazyllm_files=test_input)
        assert isinstance(result, str)
        assert len(result) > 0
        return result

    @pytest.mark.parametrize('source, init_kwargs', STT_CASES)
    def test_stt(self, source, init_kwargs):
        result = self.common_stt(source=source, **init_kwargs)
        if source == 'glm':
            assert '地铁站' in result
