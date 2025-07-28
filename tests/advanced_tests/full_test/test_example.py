import io
import os
import json
import re
import time
import httpx
import pytest
import random
from gradio_client import Client
from lazyllm.thirdparty import PIL

import lazyllm
from lazyllm.launcher import cleanup
from lazyllm.components.formatter import decode_query_with_filepaths
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_DOC_PATH


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
        r = painting(query)
        res = decode_query_with_filepaths(r)
        assert type(res) is dict
        assert "files" in res
        assert len(res['files']) == 1
        image = PIL.Image.open(res['files'][0])
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

    def test_rag_map_store_with_milvus_index(self):
        from examples.rag_map_store_with_milvus_index import run as rag_run
        res = rag_run('何为天道？')
        assert type(res) is str
        assert "天道" in res
        assert len(res) >= 16

class TestRagFilter(object):
    def setup_class(self):
        from examples.rag_milvus_store import ppl, documents, tmp_dir
        self.tmp_dir = tmp_dir
        self.documents = documents
        self.rag = lazyllm.ActionModule(ppl)
        self.rag.start()
        url_pattern = r'(http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+)'
        self.doc_server_addr = re.findall(url_pattern, documents.manager.url)[0]

    def test_upload_and_filter(self):
        files = [('files', ('test1.txt', io.BytesIO(b"John's house is in Beijing"), 'text/palin')),
                 ('files', ('test2.txt', io.BytesIO(b"John's house is in Shanghai"), 'text/plain'))]
        metadatas = [{"comment": "comment1"}, {"signature": "signature2"}]

        params = dict(override='true', metadatas=json.dumps(metadatas))

        url = f'{self.doc_server_addr}/upload_files'
        response = httpx.post(url, params=params, files=files, timeout=10)
        assert response.status_code == 200 and response.json().get('code') == 200, response.json()

        time.sleep(30)  # waiting for worker thread to update newly uploaded files

        res = self.rag("Where is John's house?", filters={'comment': ['comment1']})
        assert 'Beijing' in res and 'Shanghai' not in res

        res = self.rag("Where is John's house?", filters={'signature': ['signature2']})
        assert 'Shanghai' in res and 'Beijing' not in res

        store = self.documents._impl.store
        nodes = store.get_nodes(group='block')
        for node in nodes:
            if node.global_metadata[RAG_DOC_PATH].endswith('test1.txt'):
                test1_docid = node.global_metadata[RAG_DOC_ID]
            elif node.global_metadata[RAG_DOC_PATH].endswith('test2.txt'):
                test2_docid = node.global_metadata[RAG_DOC_ID]
        assert test1_docid and test2_docid

        res = self.rag("Where is John's house?", filters={RAG_DOC_ID: [test1_docid]})
        assert 'Beijing' in res and 'Shanghai' not in res

        res = self.rag("Where is John's house?", filters={RAG_DOC_ID: [test2_docid]})
        assert 'Shanghai' in res and 'Beijing' not in res
