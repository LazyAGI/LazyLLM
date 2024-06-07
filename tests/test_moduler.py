import pytest
import lazyllm

class TestFn_Moduler:

    def setup_method(self):
        # 在每个测试方法运行前执行的初始化代码
        self.base_model = 'base_model_path'
        self.target_path = 'target_path'
        self.data_path = 'data_path'

    def test_ActionModule(self):
        # 测试 ActionModule
        action_module = lazyllm.ActionModule(lambda x: x + 1)
        assert action_module(1) == 2
        assert action_module(10) == 11

    def test_UrlModule(self):
        # 测试 UrlModule
        url_module = lazyllm.UrlModule()
        assert url_module('input') == 'result from http://example.com with input'
        url_module.evalset(['input1', 'input2'])
        assert url_module.eval_result == ['result1', 'result2']

    def test_ServerModule(self):
        # 测试 ServerModule
        server_module = lazyllm.ServerModule(lambda x: x.upper())
        server_module.start()
        assert server_module('hello') == 'HELLO'
        server_module.evalset(['input1', 'input2'])
        assert server_module.eval_result == ['INPUT1', 'INPUT2']

    def test_TrainableModule(self):
        # 测试 TrainableModule
        trainable_module = lazyllm.TrainableModule(self.base_model, self.target_path)
        trainable_module.finetune_method(lazyllm.finetune.dummy)
        trainable_module.deploy_method(lazyllm.deploy.dummy)
        trainable_module.mode('finetune')
        trainable_module.trainset(self.data_path)
        trainable_module.prompt('prompt template')
        trainable_module.update()
        trainable_module.start()
        assert trainable_module('input') == 'result'
        trainable_module.evalset(['input1', 'input2'])
        assert trainable_module.eval_result == ['result1', 'result2']

    def test_WebModule(self):
        # 测试 WebModule 
        web_module = lazyllm.WebModule(lazyllm.ServerModule(lambda x: x.upper()), port=8080)
        web_module.start()
        web_module.evalset(['input1', 'input2'])
        assert web_module.eval_result == ['INPUT1', 'INPUT2']

    def test_OnlineChatModule(self):
        # 测试 OnlineChatModule
        chat_module = lazyllm.OnlineChatModule('openai_api_key')
        chat_module.finetune_method(lazyllm.finetune.dummy)
        chat_module.deploy_method(lazyllm.deploy.dummy) 
        chat_module.mode('finetune')
        chat_module.trainset(self.data_path)
        chat_module.prompt('prompt template')
        chat_module.update()
        chat_module.start()
        assert chat_module('input') == 'chatgpt result'
        chat_module.evalset(['input1', 'input2']) 
        assert chat_module.eval_result == ['chatgpt result1', 'chatgpt result2']