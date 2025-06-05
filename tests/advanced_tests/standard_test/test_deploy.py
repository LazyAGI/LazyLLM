import os
import json
import time
import pytest
import httpx
import random
from gradio_client import Client

import lazyllm
from lazyllm import deploy, globals
from lazyllm.launcher import cleanup
from lazyllm.components.formatter import encode_query_with_filepaths, decode_query_with_filepaths
from lazyllm.components.utils.file_operate import image_to_base64

@pytest.fixture()
def set_enviroment(request):
    env_key, env_var = request.param
    original_value = os.getenv(env_key, None)
    os.environ[env_key] = env_var
    lazyllm.config.refresh(env_key)
    yield
    if original_value:
        os.environ[env_key] = original_value
    else:
        os.environ.pop(env_key, None)
    lazyllm.config.refresh(env_key)

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

    def test_deploy_vllm(self):
        m = lazyllm.TrainableModule(self.model_path, '').deploy_method(deploy.vllm)
        m.evalset(self.inputs)
        m.update_server()
        m.eval()
        assert len(m.eval_result) == len(self.inputs)

    @pytest.mark.parametrize('set_enviroment',
                             [('LAZYLLM_DEFAULT_EMBEDDING_ENGINE', ''),
                              ('LAZYLLM_DEFAULT_EMBEDDING_ENGINE', 'transformers')],
                             indirect=True)
    def test_embedding(self, set_enviroment):
        m = lazyllm.TrainableModule('bge-large-zh-v1.5').deploy_method(deploy.AutoDeploy)
        m.update_server()
        res = m('你好')
        assert len(json.loads(res)) == 1024
        res = m(['你好'])
        assert len(json.loads(res)) == 1
        res = m(['你好', '世界'])
        assert len(json.loads(res)) == 2

    def test_sparse_embedding(self):
        m = lazyllm.TrainableModule('bge-m3').deploy_method((deploy.AutoDeploy, {'embed_type': 'sparse'}))
        m.update_server()
        res = m('你好')
        assert isinstance(json.loads(res), dict)
        res = m(['你好'])
        assert len(json.loads(res)) == 1
        res = m(['你好', '世界'])
        assert len(json.loads(res)) == 2

    def test_cross_modal_embedding(self):
        m = lazyllm.TrainableModule('siglip')
        m.update_server()
        res = m('你好')
        assert len(json.loads(res)) == 1152
        res = m(['你好'])
        assert len(json.loads(res)) == 1
        res = m(['你好', '世界'])
        assert len(json.loads(res)) == 2

        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_path = os.path.join(lazyllm.config['data_path'], "ci_data/ji.jpg")
        image_base64, mime = image_to_base64(image_path)
        image_base64 = f'data:{mime};base64,{image_base64}'
        res = m(image_url, modality='image')
        assert len(json.loads(res)) == 1152
        res = m([image_url], modality='image')
        assert len(json.loads(res)) == 1
        res = m([image_url, image_base64], modality='image')
        assert len(json.loads(res)) == 2

    def test_sd3(self):
        m = lazyllm.TrainableModule('stable-diffusion-3-medium')
        m.update_server()
        r = m('a little cat')
        res = decode_query_with_filepaths(r)
        assert "files" in res
        assert len(res['files']) == 1

    def test_musicgen(self):
        m = lazyllm.TrainableModule('musicgen-stereo-small')
        m.update_server()
        r = m('lo-fi music with a soothing melody')
        res = decode_query_with_filepaths(r)
        assert "files" in res
        assert len(res['files']) == 1

    def test_chattts(self):
        m = lazyllm.TrainableModule('ChatTTS-new')
        m.update_server()
        r = m('你好啊，很高兴认识你。')
        res = decode_query_with_filepaths(r)
        assert "files" in res
        assert len(res['files']) == 1

    def test_stt_sensevoice(self):
        chat = lazyllm.TrainableModule('sensevoicesmall')
        m = lazyllm.ServerModule(chat)
        m.update_server()
        audio_path = os.path.join(lazyllm.config['data_path'], 'ci_data/shuidiaogetou.mp3')
        res = m(audio_path)
        assert '但愿人长久' in res
        res = m(encode_query_with_filepaths(files=[audio_path]))
        assert '但愿人长久' in res
        res = m(f'<lazyllm-query>{{"query":"hi","files":["{audio_path}"]}}')
        assert '但愿人长久' in res

        _, client = self.warp_into_web(m)

        def client_send(content):
            chat_history = [[content, None]]
            ans = client.predict(self.use_context,
                                 chat_history,
                                 self.stream_output,
                                 self.append_text,
                                 api_name="/_respond_stream")
            return ans
        res = client_send(audio_path)[0][-1][-1]
        assert type(res) is str
        assert '但愿人长久' in res
        res = client_send('hi')[0][-1][-1]
        assert "Only '.mp3' and '.wav' formats in the form of file paths or URLs are supported." == res

    def test_stt_bind(self):
        audio_path = os.path.join(lazyllm.config['data_path'], 'ci_data/shuidiaogetou.mp3')
        with lazyllm.pipeline() as ppl:
            ppl.m = lazyllm.TrainableModule('sensevoicesmall') | lazyllm.bind('No use inputs', lazyllm_files=ppl.input)
        m = lazyllm.ActionModule(ppl)
        m.update_server()
        res = m(audio_path)
        assert '但愿人长久' in res
        res = m([audio_path])
        assert '但愿人长久' in res
        res = m(encode_query_with_filepaths(files=[audio_path]))
        assert '但愿人长久' in res
        res = m({"query": "aha", "files": [audio_path]})
        assert '但愿人长久' in res

    def test_vlm_and_lmdeploy(self):
        chat = lazyllm.TrainableModule('Mini-InternVL-Chat-2B-V1-5')
        m = lazyllm.ServerModule(chat)
        m.update_server()
        query = '这是啥？'
        ji_path = os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg')
        pig_path = os.path.join(lazyllm.config['data_path'], 'ci_data/pig.png')

        globals['lazyllm_files'][chat._module_id] = [pig_path]
        assert '猪' in m(query)
        globals['lazyllm_files'][chat._module_id] = None
        assert '鸡' in m(f'<lazyllm-query>{{"query":"{query}","files":["{ji_path}"]}}')

        _, client = self.warp_into_web(m)
        # Add prefix 'lazyllm_img::' for client testing.
        chat_history = [['lazyllm_img::' + ji_path, None], [query, None]]
        ans = client.predict(self.use_context,
                             chat_history,
                             self.stream_output,
                             self.append_text,
                             api_name="/_respond_stream")
        res = ans[0][-1][-1]
        assert '鸡' in res
