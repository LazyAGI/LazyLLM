import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'


class TestImageEditing:
    test_image_editing_prompt = '在参考图片中的正中间添加"LazyLLM"这段英文,字体风格要和图片相同'

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
        assert file.endswith(('.png', '.jpg', '.jpeg'))

    @staticmethod
    def _test_image_file():
        data_path = lazyllm.config['data_path']
        return [os.path.join(data_path, 'ci_data/dog.png')]

    def common_image_editing(self, source, **kwargs):
        api_key = get_api_key(source)
        kwargs.setdefault('type', 'image_editing')
        img_edit = lazyllm.OnlineModule(source=source, api_key=api_key, **kwargs)
        result = img_edit(self.test_image_editing_prompt, files=self._test_image_file())
        self._check_file_result(result)
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_image_editing_auto(self):
        self.common_image_editing(source='qwen')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_image_editing_model(self):
        self.common_image_editing(source='qwen', model='qwen-image-edit-max')

    @pytest.mark.ignore_cache_on_change(get_path('doubao'))
    @pytest.mark.xfail
    def test_doubao_image_editing(self):
        self.common_image_editing(source='doubao')

    @pytest.mark.ignore_cache_on_change(get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_image_editing_auto(self):
        self.common_image_editing(source='siliconflow')

    @pytest.mark.ignore_cache_on_change(get_path('siliconflow'))
    @pytest.mark.xfail
    def test_siliconflow_image_editing_model(self):
        self.common_image_editing(source='siliconflow', model='Qwen/Qwen-Image-Edit-2509')

    def common_image_editing_interleaved(self, source, **kwargs):
        api_key = get_api_key(source)
        kwargs.setdefault('type', 'image_editing')
        img_edit = lazyllm.OnlineModule(source=source, api_key=api_key, **kwargs)
        data_path = lazyllm.config['data_path']
        image_path = os.path.join(data_path, 'ci_data/dog.png')
        interleaved_input = [
            '这是参考图片：',
            image_path,
            '在参考图片中的正中间添加"LazyLLM"这段英文,字体风格要和图片相同'
        ]
        result = img_edit(interleaved_input)
        self._check_file_result(result)
        return result

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_image_editing_interleaved(self):
        self.common_image_editing_interleaved(source='qwen')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_image_editing_interleaved_model(self):
        self.common_image_editing_interleaved(source='qwen', model='qwen-image-edit-max')

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_image_editing_interleaved_multiple_images(self):
        api_key = get_api_key('qwen')
        img_edit = lazyllm.OnlineModule(source='qwen', api_key=api_key, type='image_editing')

        data_path = lazyllm.config['data_path']
        image_path1 = os.path.join(data_path, 'ci_data/dog.png')
        image_path2 = os.path.join(data_path, 'ci_data/ji.png')
        interleaved_input = [
            '这是第一张参考图：',
            image_path1,
            '这是第二张参考图：',
            image_path2,
            '请根据上述图片生成一张新图片，在正中间添加"LazyLLM"文字'
        ]

        result = img_edit(interleaved_input)
        self._check_file_result(result)

    @pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'))
    def test_qwen_image_editing_interleaved_conflict(self):
        api_key = get_api_key('qwen')
        img_edit = lazyllm.OnlineModule(source='qwen', api_key=api_key, type='image_editing')
        data_path = lazyllm.config['data_path']
        image_path = os.path.join(data_path, 'ci_data/dog.png')
        interleaved_input = [
            '这是参考图：',
            image_path,
            '请生成图片'
        ]

        with pytest.raises(AssertionError, match='Cannot use both interleaved input'):
            img_edit(interleaved_input, files=[image_path])
