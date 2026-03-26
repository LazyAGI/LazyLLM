

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
