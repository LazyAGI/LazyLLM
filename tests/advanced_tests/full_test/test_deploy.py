import json
import time
import pytest
import httpx
import random
from gradio_client import Client

import lazyllm
from lazyllm import deploy
from lazyllm.launcher import cleanup

class TestDeploy(object):

    def setup_method(self):
        self.model_path = 'internlm2-chat-7b'
        self.inputs = ['介绍一下你自己', '李白和李清照是什么关系', '说个笑话吧']
        self.use_context = False
        self.stream_output = False
        self.append_text = False
        self.webs = []
        self.clients = []

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        while self.clients:
            client = self.clients.pop()
            client.close()
        while self.webs:
            web = self.webs.pop()
            web.stop()
        cleanup()

    def warp_into_web(self, module):
        client = None
        for _ in range(5):
            try:
                port = random.randint(10000, 30000)
                web = lazyllm.WebModule(module, port=port)
                web._work()
                time.sleep(2)
            except AssertionError as e:
                # Port is occupied
                if 'occupied' in e:
                    continue
                else:
                    raise e
            try:
                client = Client(web.url, download_files=web.cach_path)
                break
            except httpx.ConnectError:
                continue
        assert client, "Unable to create client"
        self.webs.append(web)
        self.clients.append(client)
        return web, client

    def test_deploy_lightllm(self):
        m = lazyllm.TrainableModule(self.model_path, '').deploy_method(deploy.lightllm)
        m.evalset(self.inputs)
        m.update_server()
        m.eval()
        assert len(m.eval_result) == len(self.inputs)

    def test_deploy_auto(self):
        m = lazyllm.TrainableModule(self.model_path, '').deploy_method(deploy.AutoDeploy)
        m.evalset(self.inputs)
        m.update_server()
        m.eval()
        assert len(m.eval_result) == len(self.inputs)

    def test_deploy_auto_without_calling_method(self):
        m = lazyllm.TrainableModule(self.model_path, '')
        m.evalset(self.inputs)
        m.update_server()
        m.eval()
        assert len(m.eval_result) == len(self.inputs)

    def test_bark(self):
        m = lazyllm.TrainableModule('bark')
        m.update_server()
        res = m('你好啊，很高兴认识你。')
        assert "sounds" in json.loads(res)

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
