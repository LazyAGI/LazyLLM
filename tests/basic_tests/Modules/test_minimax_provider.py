import pytest


class TestMinimaxChatDefaults:
    '''Unit tests for MinimaxChat default configuration.'''

    def test_default_base_url(self):
        import inspect
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxChat
        sig = inspect.signature(MinimaxChat.__init__)
        assert sig.parameters['base_url'].default == 'https://api.minimax.io/v1/'

    def test_default_model(self):
        import inspect
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxChat
        sig = inspect.signature(MinimaxChat.__init__)
        assert sig.parameters['model'].default == 'MiniMax-M2.7'


class TestMinimaxText2ImageDefaults:
    '''Unit tests for MinimaxText2Image default configuration.'''

    def test_default_url(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxText2Image
        import inspect
        sig = inspect.signature(MinimaxText2Image.__init__)
        assert sig.parameters['url'].default == 'https://api.minimax.io/v1/'

    def test_default_model_name(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxText2Image
        assert MinimaxText2Image.MODEL_NAME == 'image-01'


class TestMinimaxTTSDefaults:
    '''Unit tests for MinimaxTTS default configuration.'''

    def test_default_base_url(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxTTS
        import inspect
        sig = inspect.signature(MinimaxTTS.__init__)
        assert sig.parameters['base_url'].default == 'https://api.minimax.io/v1/'

    def test_default_model_name(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxTTS
        assert MinimaxTTS.MODEL_NAME == 'speech-2.8-hd'


class TestMinimaxEmbedDefaults:
    '''Unit tests for MinimaxEmbed default configuration.'''

    def test_default_embed_url(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxEmbed
        import inspect
        sig = inspect.signature(MinimaxEmbed.__init__)
        assert sig.parameters['embed_url'].default == 'https://api.minimax.io/v1/embeddings'

    def test_default_model_name(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxEmbed
        import inspect
        sig = inspect.signature(MinimaxEmbed.__init__)
        assert sig.parameters['embed_model_name'].default == 'embo-01'


class TestMinimaxEmbedDataFormat:
    '''Unit tests for MinimaxEmbed custom data encapsulation and parsing.'''

    def _make_embed(self):
        from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxEmbed
        embed = object.__new__(MinimaxEmbed)
        embed._embed_model_name = 'embo-01'
        embed._batch_size = 16
        return embed

    def test_encapsulated_data_single_string(self):
        embed = self._make_embed()
        data = embed._encapsulated_data('hello world')
        assert isinstance(data, dict)
        assert data['model'] == 'embo-01'
        assert data['texts'] == ['hello world']
        assert data['type'] == 'db'

    def test_encapsulated_data_query_type(self):
        embed = self._make_embed()
        data = embed._encapsulated_data('search query', type='query')
        assert data['type'] == 'query'

    def test_encapsulated_data_list(self):
        embed = self._make_embed()
        data = embed._encapsulated_data(['text1', 'text2'])
        assert isinstance(data, dict)
        assert data['texts'] == ['text1', 'text2']

    def test_encapsulated_data_large_batch(self):
        embed = self._make_embed()
        embed._batch_size = 2
        texts = ['a', 'b', 'c', 'd', 'e']
        data = embed._encapsulated_data(texts)
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]['texts'] == ['a', 'b']
        assert data[1]['texts'] == ['c', 'd']
        assert data[2]['texts'] == ['e']

    def test_parse_response_single(self):
        embed = self._make_embed()
        response = {'vectors': [[0.1, 0.2, 0.3]], 'total_tokens': 5}
        result = embed._parse_response(response, input='hello')
        assert result == [0.1, 0.2, 0.3]

    def test_parse_response_list(self):
        embed = self._make_embed()
        response = {'vectors': [[0.1, 0.2], [0.3, 0.4]], 'total_tokens': 10}
        result = embed._parse_response(response, input=['a', 'b'])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_parse_response_empty_raises(self):
        embed = self._make_embed()
        response = {'vectors': [], 'total_tokens': 0}
        with pytest.raises(Exception, match='no embedding vectors received'):
            embed._parse_response(response, input='hello')


class TestMinimaxModelMapping:
    '''Unit tests for MiniMax model type classification.'''

    def test_minimax_llm_models(self):
        from lazyllm.module.llms.onlinemodule.map_model_type import get_model_type
        for model in ['minimax-m2.7', 'minimax-m2.7-highspeed', 'minimax-m2.5',
                      'minimax-m2.5-highspeed', 'minimax-m2.1', 'minimax-m2', 'minimax-m1']:
            assert get_model_type(model) == 'llm', f'{model} should be classified as llm'

    def test_minimax_image_model(self):
        from lazyllm.module.llms.onlinemodule.map_model_type import get_model_type
        assert get_model_type('image-01') == 'sd'

    def test_minimax_tts_models(self):
        from lazyllm.module.llms.onlinemodule.map_model_type import get_model_type
        for model in ['speech-2.8-hd', 'speech-2.8-turbo', 'speech-2.6-hd']:
            assert get_model_type(model) == 'tts', f'{model} should be classified as tts'

    def test_minimax_embed_model(self):
        from lazyllm.module.llms.onlinemodule.map_model_type import get_model_type
        assert get_model_type('embo-01') == 'embed'
