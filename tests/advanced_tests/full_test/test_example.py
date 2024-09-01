import os
import time
import json
import httpx
import pytest
import random
import base64
from gradio_client import Client
from PIL import Image
from io import BytesIO

import lazyllm
from lazyllm.launcher import cleanup


class TestExamples(object):

    def setup_method(self):
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

    def test_chat(self):
        from examples.chatbot import chat
        chat.start()
        query = "请原样英文输出：Hello world."
        res = chat(query)
        assert res == 'Hello world.'

        # test chat warpped in web
        _, client = self.warp_into_web(chat)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        assert ans[0][-1][-1] == 'Hello world.'

    def test_story(self):
        from examples.story import ppl
        story = lazyllm.ActionModule(ppl)
        story.start()
        query = "我的妈妈"
        res = story(query)
        assert type(res) is str
        assert len(res) >= 1024

        # test story warpped in web
        _, client = self.warp_into_web(story)
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
        from examples.rag import ppl
        rag = lazyllm.ActionModule(ppl)
        rag.start()
        query = "何为天道？"
        res = rag(query)
        assert type(res) is str
        assert "天道" in res
        assert len(res) >= 16

        # test rag warpped in web
        _, client = self.warp_into_web(rag)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        res = ans[0][-1][-1]
        assert type(res) is str
        assert "天道" in res
        assert len(res) >= 16

    def test_painting(self):
        from examples.painting import ppl
        painting = lazyllm.ActionModule(ppl)
        painting.start()
        query = "画只可爱的小猪"
        res = painting(query)
        assert type(res) is str
        imgs_bs64 = json.loads(res)
        assert "images_base64" in imgs_bs64
        image = Image.open(BytesIO(base64.b64decode(imgs_bs64["images_base64"][0])))
        assert image.size == (1024, 1024)

        # test painting warpped in web
        _, client = self.warp_into_web(painting)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        image_path = ans[0][0][-1]['value']
        assert os.path.isfile(image_path)
