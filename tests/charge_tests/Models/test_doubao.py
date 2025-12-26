import os
import pytest
import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths


@pytest.fixture
def api_key():
    return lazyllm.config['doubao_api_key']


# class TestOnlineChat(object):
#     def test_online_chat(self, api_key):
#         llm = lazyllm.OnlineChatModule(source='doubao', api_key=api_key)
#         response = llm('你好，介绍自己')
#         assert response is not None
#         assert isinstance(response, str)
#         assert len(response) > 0


class TestMultiModal(object):
    def setup_method(self):
        self.test_text = '你好，这是一个测试。'
        self.test_image_prompt = '画一只动漫风格的懒懒猫'
        self.test_image_edit_prompt = '在参考图片中的正中间添加"LazyLLM"这段英文,字体风格要和图片相同'

    def _check_file_result(self, result, format):
        assert result is not None
        assert isinstance(result, str)
        assert result.startswith('<lazyllm-query>')

        decoded = decode_query_with_filepaths(result)
        assert 'files' in decoded
        assert len(decoded['files']) > 0

        file_path = decoded['files'][0]
        assert os.path.exists(file_path)
        suffix = ('.png', '.jpg', '.jpeg') if format == 'image' else ('.wav', '.mp3', '.flac')
        assert file_path.endswith(suffix)

    def test_online_text2image(self, api_key):
        text2image = lazyllm.OnlineMultiModalModule(source='doubao', function='text2image', api_key=api_key)
        result = text2image(self.test_image_prompt)
        self._check_file_result(result, format='image')

    def test_online_text2image_edit(self, api_key):
        # png格式文件
        image_url = "https://bizyair-prod.oss-cn-shanghai.aliyuncs.com/outputs%2Fe2f95819-2703-46f1-bfad-b0f7d8136cf3_c7ba266fa220eec3a1be65cd8e1f777e_ComfyUI_81bce3ba_00001_.png?OSSAccessKeyId=LTAI5tPza7RAEKed35dCML5U&Expires=1766749359&Signature=SSvc%2F%2FYu8z1iUMqPRFH6KyaFJHI%3D"
        text2image_edit = lazyllm.OnlineMultiModalModule(source='doubao', model='doubao-seedream-4-5-251128', function='text2image', api_key=api_key, image_edit=True)
        result = text2image_edit(self.test_image_edit_prompt,files=[image_url])
        self._check_file_result(result, format='image')
    
