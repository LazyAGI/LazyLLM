import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'
QWEN_PATH = 'lazyllm/module/llms/onlinemodule/supplier/qwen.py'
DOUBAO_PATH = 'lazyllm/module/llms/onlinemodule/supplier/doubao.py'
SILICONFLOW_PATH = 'lazyllm/module/llms/onlinemodule/supplier/siliconflow.py'

pytestmark = pytest.mark.model_connectivity_test

IMAGE_EDITING_CASES = [
    pytest.param(
        'siliconflow',
        {'model': 'Qwen/Qwen-Image-Edit-2509'},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, SILICONFLOW_PATH),
        id='siliconflow_model',
    ),
    pytest.param(
        'qwen',
        {'model': 'qwen-image-edit-plus'},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, QWEN_PATH),
        id='qwen_model',
    ),
    pytest.param(
        'doubao',
        {},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, DOUBAO_PATH),
        id='doubao',
    ),
    pytest.param(
        'siliconflow',
        {},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, SILICONFLOW_PATH),
        id='siliconflow_auto',
    ),
    pytest.param(
        'qwen',
        {},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, QWEN_PATH),
        id='qwen_auto',
    ),
]


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
        api_key = lazyllm.config[f'{source}_api_key']
        kwargs.setdefault('type', 'image_editing')
        img_edit = lazyllm.OnlineModule(source=source, api_key=api_key, **kwargs)
        result = img_edit(self.test_image_editing_prompt, files=self._test_image_file())
        self._check_file_result(result)
        return result

    @pytest.mark.parametrize('source, init_kwargs', IMAGE_EDITING_CASES)
    def test_image_editing(self, source, init_kwargs):
        self.common_image_editing(source=source, **init_kwargs)
