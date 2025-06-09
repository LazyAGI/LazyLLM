import time
import requests
import pytest

import lazyllm
import multiprocessing
from lazyllm.launcher import cleanup

class TestModule:

    def setup_method(self):
        self.base_model = 'qwen2-1.5b'
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

    def test_ServerModule_with_global(self):
        lazyllm.globals['a'] = '1'
        server_module = lazyllm.ServerModule(lambda x: x.upper() + lazyllm.globals['a'])
        server_module.start()
        assert server_module('hello') == 'HELLO1'
        lazyllm.globals['a'] = '2'
        server_module.evalset(['input1', 'input2'])
        server_module.eval()
        assert server_module.eval_result == ['INPUT12', 'INPUT22']

    def test_ServerModule_url(self):
        class Test(object):
            def test1(self, a, b):
                return a + b

            def __call__(self, a, b):
                return a * b

        s = lazyllm.ServerModule(Test()).start()
        assert s(1, 2) == 2
        assert s._call('test1', 1, 2) == 3

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

        # tm5 use tm4's url
        tm5 = lazyllm.TrainableModule(self.base_model).deploy_method(tm4._deploy_type, url=tm4._url)
        tm5.evalset(inputs)
        tm5.eval()
        assert tm5.eval_result == tm4.eval_result

        tm5.prompt(None)
        tm5.evalset(inputs)
        inputs = 'input-tm5'
        assert tm5(inputs) == res_template.format(inputs)

    def test_TrainableModule_stream(self):
        tm = lazyllm.TrainableModule(self.base_model, self.target_path, stream=True).deploy_method(lazyllm.deploy.dummy)
        assert tm._deploy_type == lazyllm.deploy.dummy
        tm.prompt(None).start()

        _ = tm('input')
        re = ''.join(lazyllm.FileSystemQueue().dequeue())
        assert re == "reply for input, and parameters is {'do_sample': False, 'temperature': 0.1}"

        sm = lazyllm.ServerModule(tm)
        sm.start()
        _ = sm('input')
        re = ''.join(lazyllm.FileSystemQueue().dequeue())
        assert re == "reply for input, and parameters is {'do_sample': False, 'temperature': 0.1}"

    def test_WebModule(self):
        def func(x):
            return 'reply ' + x
        m = lazyllm.WebModule(func)
        m.update()
        time.sleep(4)
        response = requests.get(m.url)
        assert response.status_code == 200
        m.stop()

    # for mac
    def test_WebModule_spawn(self):
        m = multiprocessing.get_start_method()
        if m != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        self.test_WebModule()
        if m != 'spawn':
            multiprocessing.set_start_method(m, force=True)

    def test_custom_module(self):
        class MyModule(lazyllm.ModuleBase):
            def forward(self, a, b):
                return a + b

        assert MyModule()(1, 2) == 3
        assert lazyllm.pipeline(MyModule)(1, 2) == 3
        assert lazyllm.pipeline(MyModule())(1, 2) == 3

        class MyModule(lazyllm.ModuleBase):
            def forward(self, a):
                return a['a'] + a['b']

        with lazyllm.parallel().sum as prl:
            prl.m1 = MyModule
            prl.m2 = MyModule()

        assert prl(dict(a=1, b=2)) == 6

        with lazyllm.warp().sum as prl:
            prl.m1 = MyModule

        assert prl([dict(a=1, b=2), dict(a=3, b=4)]) == 10
        assert prl(dict(a=1, b=2), dict(a=3, b=4)) == 10
