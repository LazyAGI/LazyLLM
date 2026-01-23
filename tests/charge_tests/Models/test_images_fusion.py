import os

import pytest

import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineMultiModalBase.py'
QWEN_PATH = 'lazyllm/module/llms/onlinemodule/supplier/qwen.py'
SILICONFLOW_PATH = 'lazyllm/module/llms/onlinemodule/supplier/siliconflow.py'

pytestmark = pytest.mark.model_connectivity_test

IMAGES_FUSION_CASES = [
    pytest.param(
        'qwen',
        {'model': 'qwen-image-edit-plus'},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, QWEN_PATH),
        id='qwen',
    ),
    pytest.param(
        'siliconflow',
        {'model': 'Qwen/Qwen-Image-Edit-2509'},
        marks=pytest.mark.ignore_cache_on_change(BASE_PATH, SILICONFLOW_PATH),
        id='siliconflow',
    ),
]


class TestImagesFusion:
    test_multi_images_fusion = '将上传的参考图片融合在一起'

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
    def _test_images_files():
        data_path = lazyllm.config['data_path']
        file = os.path.join(data_path, 'ci_data/ji.jpg')
        return [file, file]

    def common_images_fusion(self, source, **kwargs):
        api_key = lazyllm.config[f'{source}_api_key']
        kwargs.setdefault('type', 'image_editing')
        img_fusion = lazyllm.OnlineModule(source=source, api_key=api_key, **kwargs)
        result = img_fusion(self.test_multi_images_fusion, files=self._test_images_files())
        self._check_file_result(result)
        return result

    @pytest.mark.parametrize('source, init_kwargs', IMAGES_FUSION_CASES)
    def test_images_fusion(self, source, init_kwargs):
        self.common_images_fusion(source=source, **init_kwargs)
