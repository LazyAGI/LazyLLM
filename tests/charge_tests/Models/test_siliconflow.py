import os
import pytest
import lazyllm
from lazyllm.components.formatter import decode_query_with_filepaths

@pytest.fixture
def api_key():
    return lazyllm.config['siliconflow_api_key']

class TestOnlineChat(object):
    def test_online_chat(self, api_key):
        llm = lazyllm.OnlineChatModule(source='siliconflow', api_key=api_key)
        response = llm('你好，介绍自己')
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

class TestOnlineEmbedding(object):
    def test_online_embed(self, api_key):
        embed_model = lazyllm.OnlineEmbeddingModule(source='siliconflow', api_key=api_key)
        vec1 = embed_model('床前明月光')
        vec2 = embed_model(['床前明月光', '疑是地上霜'])
        assert len(vec2) == 2
        assert len(vec2[0]) == len(vec1)
        assert len(vec2[1]) == len(vec1)

    def test_online_rerank(self, api_key):
        rerank_model = lazyllm.OnlineEmbeddingModule(source='siliconflow', api_key=api_key, type='rerank')
        result = rerank_model('床前明月光', documents=['床前明月光', '疑是地上霜', '举头望明月', '低头思故乡'], top_n=4)
        indices = [item[0] for item in result]  # item[0]: index
        scores = [item[1] for item in result]  # item[1]: relevance_score
        assert len(result) == 4
        assert set(indices) == {0, 1, 2, 3}
        for item in result:
            assert 0 <= item[1] <= 1
        assert scores[0] == max(scores)

class TestMultiModal(object):
    def setup_method(self):
        self.test_text = '你好，这是一个测试。'
        self.test_image_prompt = '画一只动漫风格的懒懒猫'
        self.test_image_editing_prompt = '在参考图片中的正中间添加"LazyLLM"这段英文,字体风格要和图片相同'
        self.test_image_file ='.temp/template.png' or os.path.join(lazyllm.config['data_path'], 'ci_data/pig.png')

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

    def test_online_tts(self, api_key):
        tts = lazyllm.OnlineMultiModalModule(source='siliconflow', function='tts', api_key=api_key)
        result = tts(self.test_text, voice='fnlp/MOSS-TTSD-v0.5:anna')
        self._check_file_result(result, format='audio')

    def test_online_text2image(self, api_key):
        sd = lazyllm.OnlineMultiModalModule(source='siliconflow', function='text2image', api_key=api_key)
        result = sd(self.test_image_prompt)
        self._check_file_result(result, format='image')
    