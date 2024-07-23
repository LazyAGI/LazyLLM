import os
import time
import httpx
import pytest
import random
from gradio_client import Client

import lazyllm
from lazyllm.launcher import cleanup


class TestExamples(object):

    def setup_method(self):
        self.use_context = False
        self.stream_output = False
        self.append_text = False
        self.env_vars = [
            'LAZYLLM_OPENAI_API_KEY',
            'LAZYLLM_KIMI_API_KEY',
            'LAZYLLM_QWEN_API_KEY',
            'LAZYLLM_SENSENOVA_API_KEY',
        ]
        self.webs = []
        self.clients = []

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        env_vars = {}
        for var in self.env_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
                del os.environ[var]
                env_name = var[8:].lower()
                lazyllm.config.add(env_name.lower(), str, "", env_name)
        yield
        for var, value in env_vars.items():
            os.environ[var] = value
            env_name = var[8:]
            lazyllm.config.add(env_name.lower(), str, "", env_name)
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

    def test_chat(self):
        from examples.chatbot_online import chat
        chat.start()
        query = "不要发挥和扩展，请严格原样输出下面句子：Hello world."
        res = chat(query)
        assert res == 'Hello world.'

        # test chat warpped in web
        web, client = self.warp_into_web(chat)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        assert ans[0][-1][-1] == 'Hello world.'

    def test_story(self):
        from examples.story_online import ppl
        story = lazyllm.ActionModule(ppl)
        story.start()
        query = "我的妈妈"
        res = story(query)
        assert type(res) is str
        assert len(res) >= 1024

        # test story warpped in web
        web, client = self.warp_into_web(story)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        res = ans[0][-1][-1]
        assert type(res) is str
        assert len(res) >= 1024

    def test_rag(self):
        from examples.rag_online import ppl
        rag = lazyllm.ActionModule(ppl)
        rag.start()
        query = "何为天道？"
        res = rag(query)
        assert type(res) is str
        assert len(res) >= 16

        # test rag warpped in web
        web, client = self.warp_into_web(rag)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        res = ans[0][-1][-1]
        assert type(res) is str
        assert len(res) >= 16
