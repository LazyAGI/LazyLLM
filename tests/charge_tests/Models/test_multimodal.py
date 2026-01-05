import os
import lazyllm
import pytest
from lazyllm import config
from lazyllm.components.formatter import decode_query_with_filepaths

class TestMultiModal(object):

    def setup_method(self):
        self.test_text = '你好，这是一个测试。'
        self.test_image_prompt = '画一只动漫风格的懒懒猫'
        self.test_audio_file = os.path.join(lazyllm.config['data_path'], 'ci_data/asr_test.wav')
        self.test_audio_file_url = (
            'https://dashscope.oss-cn-beijing.aliyuncs.com/'
            'samples/audio/paraformer/hello_world_male2.wav'
        )
        self.test_image_editing_prompt = '在参考图片中的正中间添加"LazyLLM"这段英文,字体风格要和图片相同'
        # self.test_image_file = os.path.join(lazyllm.config['data_path'], 'ci_data/pig.png')
        self.test_image_file = "D:\\template.png"

    def test_online_tts(self):
        api_key = lazyllm.config['qwen_api_key']
        tts = lazyllm.OnlineMultiModalModule(source='qwen', function='tts', model='qwen-tts', api_key=api_key)
        result = tts(self.test_text)
        self._check_file_result(result, format='audio')

        if config['cache_online_module']: return
        with pytest.raises(RuntimeError, match='cosyvoice-v1 does not support multi user, don\'t set api_key'):
            tts = lazyllm.OnlineMultiModalModule(source='qwen', function='tts', model='cosyvoice-v1', api_key=api_key)
            result = tts(self.test_text)

    def test_online_stt(self):
        stt = lazyllm.OnlineMultiModalModule(source='glm', function='stt')
        result = stt(lazyllm_files=self.test_audio_file)
        assert '地铁站' in result

    def _check_file_result(self, result, format):
        assert result is not None
        assert isinstance(result, str)
        assert result.startswith('<lazyllm-query>')

        decoded = decode_query_with_filepaths(result)
        assert 'files' in decoded
        assert len(decoded['files']) > 0

        file = decoded['files'][0]
        assert os.path.exists(file)
        suffix = ('.png', '.jpg', '.jpeg') if format == 'image' else ('.wav', '.mp3', '.flac')
        assert file.endswith(suffix)

    @pytest.mark.ignore_cache_on_change('lazyllm/module/llms/onlinemodule/supplier/siliconflow.py')
    def test_online_text2image(self):
        text2image = lazyllm.OnlineMultiModalModule(source='siliconflow', function='text2image', model='Kolors')
        result = text2image(self.test_image_prompt)
        self._check_file_result(result, format='image')

    def test_online_image_editinging_siliconflow(self):
        api_key = lazyllm.config['siliconflow_api_key']
        text2image_editing = lazyllm.OnlineModule(source='siliconflow', model='Qwen/Qwen-Image-Edit', 
                                                  function='image_editing', api_key=api_key)
        result = text2image_editing(self.test_image_editing_prompt,files=[self.test_image_file])
        self._check_file_result(result, format='image')

    def test_online_image_editinging_qwen(self):
        api_key = lazyllm.config['qwen_api_key']
        text2image_editing = lazyllm.OnlineModule(source='qwen', model='qwen-image-edit-plus', 
                                                  function='image_editing', api_key=api_key)
        result = text2image_editing(self.test_image_editing_prompt,files=[self.test_image_file])
        self._check_file_result(result, format='image')

    def test_online_image_editinging_siliconflow(self):
        api_key = lazyllm.config['siliconflow_api_key']
        text2image_editing = lazyllm.OnlineMultiModalModule(source='siliconflow', model='Qwen/Qwen-Image-Edit',function='text2image', api_key=api_key)
        result = text2image_editing(self.test_image_editing_prompt,files=[self.test_image_file])
        self._check_file_result(result, format='image')

    def test_online_image_editinging_qwen(self):
        api_key = lazyllm.config['qwen_api_key']
        text2image_editing = lazyllm.OnlineMultiModalModule(source='qwen', model='qwen-image-edit',function='text2image', api_key=api_key)
        result = text2image_editing(self.test_image_editing_prompt,files=[self.test_image_file])
        self._check_file_result(result, format='image')
