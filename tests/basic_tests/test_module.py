
import time
import requests

import lazyllm
from lazyllm.launcher import cleanup

class TestFn_Module:

    def setup_method(self):
        self.base_model = 'internlm2-chat-7b'
        self.target_path = ''
        self.data_path = 'data_path'

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
        cleanup()

    def test_ServerModule(self):
        server_module = lazyllm.ServerModule(lambda x: x.upper())
        server_module.start()
        assert server_module('hello') == 'HELLO'
        server_module.evalset(['input1', 'input2'])
        server_module.eval()
        assert server_module.eval_result == ['INPUT1', 'INPUT2']

    def test_TrainableModule(self):
        trainable_module = lazyllm.TrainableModule(self.base_model, self.target_path)
        trainable_module.finetune_method(lazyllm.finetune.dummy)
        trainable_module.deploy_method(lazyllm.deploy.dummy)
        trainable_module.mode('finetune')
        trainable_module.trainset(self.data_path)
        trainable_module.prompt(prompt=None)
        trainable_module.update()
        res_template = "reply for {}, and parameters is {{'do_sample': False, 'temperature': 0.1}}"
        inputs = 'input'
        assert trainable_module(inputs) == res_template.format(inputs)
        inputs = ['input1', 'input2']
        trainable_module.evalset(['input1', 'input2'])
        trainable_module.eval()
        assert trainable_module.eval_result == [res_template.format(x) for x in inputs]

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
