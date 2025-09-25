import os
import time
import httpx
import pytest
import random
from gradio_client import Client

import lazyllm
from lazyllm import config
from lazyllm.launcher import cleanup
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.prompter import ChatPrompter


class TestExamples(object):

    def setup_method(self):
        self.use_context = False
        self.stream_output = False
        self.append_text = False
        self.env_vars = [
            'LAZYLLM_OPENAI_API_KEY',
            'LAZYLLM_KIMI_API_KEY',
            'LAZYLLM_SENSENOVA_API_KEY',
            'LAZYLLM_DOUBAO_API_KEY',
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

    def warp_into_web(self, module, file_target=None):
        client = None
        for _ in range(5):
            try:
                port = random.randint(10000, 30000)
                web = lazyllm.WebModule(module, port=port, files_target=file_target)
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

        # test chat warpped in web
        web, client = self.warp_into_web(chat)
        chat_history = [[query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        assert ans[0][-1][-1] == 'Hello world.'

    def test_vl_chat(self):
        from examples.multimodal_chatbot_online import chat
        chat.start()
        chat.prompt(ChatPrompter({
            "system": "你是一个图片识别专家，请根据图片回答问题",
            "user": "回答问题{query}"
        }))
        query = "图中的动物是猫吗？输出Y代表是，N代表不是。"
        file_path = os.path.join(lazyllm.config['data_path'], "ci_data/ji.jpg")
        inputs = encode_query_with_filepaths(query, [file_path])
        res = chat(inputs)
        assert 'N' in res

        # test vl chat warpped in web
        web, client = self.warp_into_web(chat, file_target=chat)
        chat_history = [[f"lazyllm_img::{file_path}", None], [query, None]]
        ans = client.predict(self.use_context, chat_history, self.stream_output, self.append_text,
                             api_name="/_respond_stream")
        assert 'N' in ans[0][-1][-1]

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

        # test pipeline wrapped into iterator
        if not config['cache_online_module']:
            from lazyllm.tools.common import StreamCallHelper
            flow_iterator = StreamCallHelper(ppl, interval=0.01)
            res_list = list(flow_iterator(query))
            assert isinstance(res_list, list) and len(res_list) > 1

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

@pytest.fixture()
def requestOnlineChatModule(request):
    params = request.param if hasattr(request, "param") else {}
    source = params.get("source", None)
    query = params.get("query", "")
    print(f"\nStarting test 【{source}】 Module.")
    chat = lazyllm.OnlineChatModule(source=source)
    res = chat(query)
    yield res
    print(f"\n【{source}】Module test done.")

query = "不要发挥和扩展，请严格原样输出下面句子：Hello world."

class TestOnlineChatModule(object):
    @pytest.mark.parametrize("requestOnlineChatModule",
                             [{"source": "sensenova", "query": query},
                              {"source": "glm", "query": query},
                              {"source": "kimi", "query": query},
                              {"source": "qwen", "query": query},
                              {"source": "doubao", "query": query}],
                             indirect=True)
    def test_online_chat(self, requestOnlineChatModule):
        res = requestOnlineChatModule
        assert res == 'Hello world.'
