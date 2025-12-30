import json
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

    def test_OnlineModule_forward_override(self, monkeypatch):
        request_records = []

        class DummyResponse:
            def __init__(self, url, payload, text=None):
                self.url = url
                self._payload = payload
                self.status_code = 200
                self.text = text or json.dumps(payload)

            def json(self):
                return self._payload

            def iter_content(self, chunk_size=None):
                yield self.text.encode()

            def iter_lines(self):
                yield self.text.encode()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_post(url, json=None, headers=None, stream=False, proxies=None, timeout=None):
            request_records.append({'url': url, 'payload': json})
            if 'chat/completions' in url:
                body = {'choices': [{'message': {'content': 'ok'}}],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
                return DummyResponse(url, body, text=jsonlib.dumps(body))
            else:
                return DummyResponse(url, {'data': [{'embedding': [0.1, 0.2]}]})

        jsonlib = json
        monkeypatch.setattr('requests.post', fake_post)

        chat = lazyllm.OnlineModule(source='openai', 
                                    url='http://base/v1/', 
                                    model='base_model', 
                                    api_key='dummy_key')
        embed = lazyllm.OnlineModule(type='embed', 
                                     source='openai', 
                                     url='http://base-embed/v1/', 
                                     model='base_embed_model', 
                                     api_key='dummy_key')

        chat('hello')
        assert request_records[-1]['url'].startswith('http://base/v1/')
        assert request_records[-1]['payload']['model'] == 'base_model'

        chat('runtime', model='override_model', url='http://runtime-chat/v1/')
        assert request_records[-1]['url'].startswith('http://runtime-chat/v1/')
        assert request_records[-1]['payload']['model'] == 'override_model'

        chat('again')
        assert request_records[-1]['url'].startswith('http://base/v1/')
        assert request_records[-1]['payload']['model'] == 'base_model'

        embed('text')
        assert request_records[-1]['url'].startswith('http://base-embed/v1/')
        assert request_records[-1]['payload']['model'] == embed._embed_model_name

        embed('runtime', model='embed_override', url='http://runtime-embed/v1/')
        assert request_records[-1]['url'].startswith('http://runtime-embed/v1/')
        assert request_records[-1]['payload']['model'] == 'embed_override'

        embed('final')
        assert request_records[-1]['url'].startswith('http://base-embed/v1/')
        assert request_records[-1]['payload']['model'] == embed._embed_model_name

    def test_OnlineMultiModal_forward_override(self):
        class DummyMulti(lazyllm.module.OnlineMultiModalBase):
            def __init__(self):
                super().__init__('DUMMY', model_name='whisper-1', api_key='dummy', base_url='http://base')
                self.records = []

            def _forward(self, input: str = None, model: str = None, url: str = None, **kwargs):
                model = model or self._model_name
                url = url or self._base_url
                self.records.append((model, url, input))
                return model + ', ' + input

        dummy = DummyMulti()
        assert dummy('hello') == 'whisper-1, hello'
        assert dummy.records[-1] == ('whisper-1', 'http://base', 'hello')

        assert dummy('new', model='glm-asr', url='http://override') == 'glm-asr, new'
        assert dummy.records[-1] == ('glm-asr', 'http://override', 'new')

        assert dummy('final') == 'whisper-1, final'
        assert dummy.records[-1] == ('whisper-1', 'http://base', 'final')