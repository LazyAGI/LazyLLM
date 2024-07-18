
import time
import requests
import pytest

import lazyllm
from lazyllm.launcher import cleanup

class TestModule:

    def setup_method(self):
        self.base_model = 'internlm2-chat-7b'
        self.target_path = ''
        self.data_path = 'data_path'

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        cleanup()

    def test_ActionModule(self):
        action_module = lazyllm.ActionModule(lambda x: x + 1)
        assert action_module(1) == 2
        assert action_module(10) == 11

    def test_UrlModule(self):
        def func(x):
            return str(x) + ' after'
        # Generate accessible URL service:
        m1 = lazyllm.ServerModule(func)
        m1.update()

        m2 = lazyllm.UrlModule(url=m1._url)
        assert m2._url == m1._url
        m2.evalset([1, 'hi'])
        m2.update()
        assert m2.eval_result == ['1 after', 'hi after']

    def test_ServerModule(self):
        server_module = lazyllm.ServerModule(lambda x: x.upper())
        server_module.start()
        assert server_module('hello') == 'HELLO'
        server_module.evalset(['input1', 'input2'])
        server_module.eval()
        assert server_module.eval_result == ['INPUT1', 'INPUT2']

    def test_TrainableModule(self):
        tm1 = lazyllm.TrainableModule(self.base_model, self.target_path)
        tm2 = tm1.share()
        # tm1 and tm2 all use: ChatPrompter
        assert tm1._prompt == tm2._prompt
        tm1.finetune_method(lazyllm.finetune.dummy)\
            .deploy_method(lazyllm.deploy.dummy)\
            .mode('finetune').trainset(self.data_path)
        tm1.prompt(prompt=None)
        # tm1 use EmptyPrompter, tm2 use: ChatPrompter
        assert tm1._prompt != tm2._prompt
        assert type(tm2._prompt) is lazyllm.ChatPrompter
        assert type(tm1._prompt) is lazyllm.prompter.EmptyPrompter
        tm1.update()

        res_template = "reply for {}, and parameters is {{'do_sample': False, 'temperature': 0.1}}"
        inputs = 'input'
        assert tm1(inputs) == res_template.format(inputs)

        inputs = ['input1', 'input2']
        tm1.evalset(inputs)
        tm1.eval()
        assert tm1.eval_result == [res_template.format(x) for x in inputs]
        tm2.evalset(inputs)
        tm2.eval()
        assert tm2.eval_result == ["\n, and parameters is {'do_sample': False, 'temperature': 0.1}"] * 2

        tm3 = tm1.share()
        # tm1 and tm3 use same: EmptyPrompter
        assert type(tm3._prompt) is lazyllm.prompter.EmptyPrompter
        assert tm1._prompt == tm3._prompt
        tm3.evalset(inputs)
        tm3.eval()
        assert tm1.eval_result == tm3.eval_result

        tm4 = tm2.share()
        # tm2 and tm4 use same: ChatPrompter
        assert type(tm4._prompt) is lazyllm.ChatPrompter
        assert tm4._prompt == tm2._prompt
        tm4.evalset(inputs)
        tm4.eval()
        assert tm4.eval_result == tm2.eval_result

        # tm2 use EmptyPrompter, tm4 use: ChatPrompter
        tm2.prompt(prompt=None)
        assert tm2._prompt != tm4._prompt
        assert type(tm4._prompt) is lazyllm.ChatPrompter
        assert type(tm2._prompt) is lazyllm.prompter.EmptyPrompter

    def test_WebModule(self):
        def func(x):
            return 'reply ' + x
        m = lazyllm.WebModule(func)
        m.update()
        time.sleep(4)
        assert m.p.is_alive()
        response = requests.get(m.url)
        assert response.status_code == 200
        m.stop()

    def test_AutoModel(self):
        # No model_name and key
        chat = lazyllm.AutoModel()
        assert isinstance(chat, lazyllm.TrainableModule)
        # No model_name, but set key
        lazyllm.config.add("openai_api_key", str, "123", "OPENAI_API_KEY")
        chat = lazyllm.AutoModel()
        assert isinstance(chat, lazyllm.OnlineChatModule)
        # set model_name and key
        chat = lazyllm.AutoModel('internlm2-chat-7b')
        assert isinstance(chat, lazyllm.TrainableModule)
