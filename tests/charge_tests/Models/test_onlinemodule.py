import lazyllm


class TestOnlineModule(object):

    def test_OnlineModule_init(self):

        def assert_cases(expected_cls, cases):
            for kwargs in cases:
                module = lazyllm.OnlineModule(**kwargs)
                assert isinstance(module, expected_cls)

        chat_cases = [
            {},
            {'type': 'llm'},
            {'model': 'DeepSeek-V3'},
            {'type': 'vlm'},
            {'model': 'SenseNova-V6-5-Pro'},
        ]
        assert_cases(lazyllm.module.OnlineChatModuleBase, chat_cases)

        embed_cases = [
            {'type': 'embed'},
            {'model': 'nova-embedding-stable'},
            {'type': 'cross_modal_embed'},
            {'model': 'qwen2.5-vl-embedding'},
            {'type': 'rerank'},
            {'model': 'rerank'},
        ]
        assert_cases(lazyllm.module.OnlineEmbeddingModuleBase, embed_cases)

        multimodal_cases = [
            {'type': 'stt'},
            {'model': 'qwen-audio-turbo'},
            {'type': 'tts'},
            {'model': 'qwen-tts'},
            {'type': 'sd'},
            {'model': 'qwen-image'},
        ]
        assert_cases(lazyllm.module.OnlineMultiModalModule, multimodal_cases)

    def test_OnlineModule_inherit_register(self):
        from lazyllm.module import OnlineChatModuleBase

        class TestChat(OnlineChatModuleBase):
            __lazyllm_registry_key__ = 'test'

            def __init__(self, base_url, model, api_key, stream, return_trace, skip_auth, **kw):
                super().__init__(
                    self,
                    # model_series='Test',
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model,
                    stream=stream,
                    return_trace=return_trace,
                    skip_auth=skip_auth,
                    **kw
                )

        assert lazyllm.online.chat.test == TestChat
        for key in ['chat', 'embed', 'rerank', 'stt', 'text2image', 'tts']:
            assert 'qwen' in lazyllm.online[key].keys()
    
    def test_OnlineModule_config_register(self):
        for api_key in ['openai_api_key', 'qwen_model_name', 'sensenova_secret_key']:
            assert api_key in lazyllm.config.get_all_configs()

    def test_OnlineChat_forward_override(self, monkeypatch):
        class DummyChat(lazyllm.module.OnlineChatModuleBase):
            def __init__(self, base_url: str, model: str, api_key: str, stream: bool = False, **kw):
                super().__init__(model_series='DUMMY', api_key=api_key, base_url=base_url,
                                 model_name=model, stream=stream, **kw)

            def _get_system_prompt(self):
                return ''

            def forward(self, __input=None, *, url: str = None, model: str = None, **kw):
                runtime_base_url = url or kw.pop('base_url', None)
                runtime_url = self._get_chat_url(runtime_base_url) if runtime_base_url else self._chat_url
                runtime_model = model or kw.pop('model_name', None) or self._model_name
                return runtime_model + ', ' + __input + ', ' + runtime_url

        monkeypatch.setitem(lazyllm.online.chat, 'openai', DummyChat)

        chat = lazyllm.OnlineModule(source='openai',
                                    url='http://base/v1/',
                                    model='base_model',
                                    api_key='dummy_key')

        res = chat('hello')
        assert res == 'base_model, hello, http://base/v1/chat/completions'

        res = chat('runtime', model='override_model', url='http://runtime-chat/v1/')
        assert res == 'override_model, runtime, http://runtime-chat/v1/chat/completions'

        res = chat('again')
        assert res == 'base_model, again, http://base/v1/chat/completions'

    def test_OnlineEmbedding_forward_override(self, monkeypatch):
        class DummyEmbed(lazyllm.module.OnlineEmbeddingModuleBase):
            def __init__(self, embed_url: str, embed_model_name: str, api_key: str, **kw):
                super().__init__(model_series='DUMMY', embed_url=embed_url, api_key=api_key,
                                 embed_model_name=embed_model_name, **kw)

            def forward(self, input, url: str = None, model: str = None, **kwargs):
                runtime_url = url or kwargs.pop('base_url', kwargs.pop('embed_url', None)) or self._embed_url
                runtime_model = model or kwargs.pop('model_name', kwargs.pop('embed_model_name', None)) \
                    or self._embed_model_name
                return runtime_model + ', ' + input + ', ' + runtime_url

        monkeypatch.setitem(lazyllm.online.embed, 'openai', DummyEmbed)

        embed = lazyllm.OnlineModule(type='embed',
                                     source='openai',
                                     url='http://base-embed/v1/',
                                     model='base_embed_model',
                                     api_key='dummy_key')

        res = embed('text')
        assert res == 'base_embed_model, text, http://base-embed/v1/'

        res = embed('runtime', model='embed_override', url='http://runtime-embed/v1/')
        assert res == 'embed_override, runtime, http://runtime-embed/v1/'

        res = embed('final')
        assert res == 'base_embed_model, final, http://base-embed/v1/'

    def test_OnlineMultiModal_forward_override(self):
        class DummyMulti(lazyllm.module.OnlineMultiModalBase):
            def __init__(self):
                super().__init__('DUMMY', model_name='whisper-1', api_key='dummy', base_url='http://base')

            def _forward(self, input: str = None, model: str = None, url: str = None, **kwargs):
                model = model or self._model_name
                url = url or self._base_url
                return model + ', ' + input + ', ' + url

        dummy = DummyMulti()
        dummy.use_cache(False)
        assert dummy('hello') == 'whisper-1, hello, http://base'
        assert dummy('new', model='glm-asr', url='http://override') == 'glm-asr, new, http://override'
        assert dummy('final') == 'whisper-1, final, http://base'
