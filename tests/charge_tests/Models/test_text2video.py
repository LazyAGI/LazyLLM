import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'


class TestText2Video:
    test_video_prompt = '一只小狗在草地上跳一段简短舞蹈'

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
        assert file.endswith(('.mp4', '.webm', '.mov'))

    def common_text2video(self, source, type='text2video', call_kwargs=None, **kwargs):
        api_key = get_api_key(source)
        t2v = lazyllm.OnlineMultiModalModule(source=source, type=type, api_key=api_key, **kwargs)
        # Keep duration short in charge tests to reduce cost.
        call_kwargs = {'duration': 2, 'resolution': '480p', 'ratio': '16:9', **(call_kwargs or {})}
        result = t2v(self.test_video_prompt, **call_kwargs)
        self._check_file_result(result)
        return result

    @staticmethod
    def _test_image_file():
        data_path = lazyllm.config['data_path']
        return [os.path.join(data_path, 'ci_data/dog.png')]

    def common_image2video(self, source, type='text2video', call_kwargs=None, **kwargs):
        api_key = get_api_key(source)
        i2v = lazyllm.OnlineMultiModalModule(source=source, type=type, api_key=api_key, **kwargs)
        call_kwargs = {'duration': 2, 'resolution': '480p', 'ratio': '16:9', **(call_kwargs or {})}
        result = i2v(self.test_video_prompt, files=self._test_image_file(), **call_kwargs)
        self._check_file_result(result)
        return result

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.skip(reason='Doubao text2video is skipped by default (paid / slow)')
    def test_doubao_text2video(self):
        self.common_text2video(
            source='doubao',
            model='doubao-seedance-1-0-pro-fast-251015',
        )

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.skip(reason='Doubao text2video is skipped by default (paid / slow)')
    def test_doubao_image2video(self):
        self.common_image2video(
            source='doubao',
            model='doubao-seedance-1-0-pro-fast-251015',
        )

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.skip(reason='Doubao text2video is skipped by default (paid / slow)')
    def test_doubao_text2video_pro(self):
        self.common_text2video(
            source='doubao',
            model='doubao-seedance-1-0-pro-250528',
        )
