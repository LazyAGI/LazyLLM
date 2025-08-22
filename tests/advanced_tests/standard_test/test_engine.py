import os
import time
import pytest
import requests
from urllib.parse import urlparse
import uuid

import lazyllm
from lazyllm.engine import LightEngine
from lazyllm.tools.infer_service.serve import InferServer


class TestEngine(object):
    # This test requires 4 GPUs and takes about 4 minutes to execute, skip this test to save time.
    def setup_method(self):
        self.infer_server = lazyllm.ServerModule(InferServer(), launcher=lazyllm.launcher.EmptyLauncher(sync=False))
        self.infer_server.start()()
        parsed_url = urlparse(self.infer_server._url)
        self.infer_server_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        token = '123'
        self.headers = {"token": token}

    def teardown_method(self):
        self.infer_server.stop()

    def _test_vqa(self):
        resource = [dict(id='0', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', history=[], audio=True))]
        node = [dict(id='1', kind='VQA', name='vqa', args=dict(base_model='Mini-InternVL-Chat-2B-V1-5', type='local'))]
        edge = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        engine.start(node, edge, resource)

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        LightEngine().reset()
        lazyllm.FileSystemQueue().dequeue()
        lazyllm.FileSystemQueue(klass="lazy_trace").dequeue()

    def test_http(self):
        nodes = [
            dict(
                id="1",
                kind="HTTP",
                name="visit_sensetime",
                args=dict(
                    method="GET",
                    url="https://www.sensetime.com/cn",
                    api_key=None,
                    headers=None,
                    params=None,
                    body=None,
                )
            )
        ]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        ret = engine.run(gid)
        assert '商汤科技' in ret['content']

    def test_multimedia(self):
        painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'  # noqa E501
        musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'  # noqa E501
        translator_prompt = 'Now you are a master of translation prompts, capable of converting any Chinese content entered by the user into English translation prompts. In this task, you need to convert any input content into English translation prompts, and you can enrich and expand the prompt content.'  # noqa E501

        resources = [dict(id='llm', kind='LLM', name='base', args=dict(base_model='internlm2-chat-7b', type='local')),
                     dict(id='file-resource', kind='File', name='file', args=dict(id='file-resource')),
                     dict(id='vqa', kind='VQA', name='vqa',
                          args=dict(base_model='Mini-InternVL-Chat-2B-V1-5', type='local')),
                     dict(id='web', kind='web', name='web', args=dict(port=None, title='多模态聊天机器人', audio=True))]

        nodes1 = [
            dict(id='2', kind='SharedModel', name='draw_prompt', args=dict(llm='llm', prompt=painter_prompt, cls='llm')),
            dict(id='3', kind='SD', name='sd', args=dict(base_model='stable-diffusion-3-medium', type='local')),
            dict(id='5', kind='SharedModel', name='vqa1', args=dict(llm='vqa', cls='vqa')),
            dict(id='6', kind='JoinFormatter', name='merge_sd_vqa2', args=dict(type='file')),
        ]
        edges1 = [
            dict(iid='__start__', oid='2'), dict(iid='6', oid='__end__'), dict(iid="2", oid="3"),
            dict(constant='描述图片', oid="5"), dict(iid="3", oid="5"), dict(iid="3", oid="6"), dict(iid="5", oid="6"),
        ]

        nodes = [dict(id='7', kind='STT', name='stt', args=dict(base_model='SenseVoiceSmall', type='local')),
                 dict(id='8', kind='Intention', name='intent', args=dict(base_model='llm', nodes={
                     'Drawing': dict(id='9', kind='SubGraph', name='draw_vqa', args=dict(nodes=nodes1, edges=edges1)),
                     'Translate': dict(id='10', kind='SharedModel', name='translate_prompt',
                                       args=dict(llm='llm', prompt=translator_prompt, cls='llm')),
                     'Generate Music': [dict(id='11', kind='SharedModel', name='translate',
                                             args=dict(llm='llm', prompt=musician_prompt, cls='stt')),
                                        dict(id='12', kind='TTS', name='music',
                                             args=dict(base_model='musicgen-small', type='local'))],
                     'Image Question Answering': dict(id='13', kind='SharedModel', name='vqa2',
                                                      args=dict(llm='vqa', file_resource_id='file-resource', cls='vqa')),
                     'Chat': dict(id='14', kind='SharedModel', name='chat', args=dict(llm='llm', cls='llm'))}))]
        edges = [dict(iid="__start__", oid="7"), dict(iid="7", oid="8"), dict(iid="8", oid="__end__")]

        engine = LightEngine()
        gid = engine.start(nodes, edges, resources)

        r = engine.run(gid, '画个猪')
        assert '.png' in r

        r = engine.run(gid, '翻译：我喜欢敲代码。')
        assert 'code' in r or 'coding' in r

        r = engine.run(gid, "", _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/draw_pig.mp3'))
        assert '.png' in r

        r = engine.run(gid, "这张图片描述的是什么？", _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

        r = engine.run(gid, "这张图片描述的是什么？",
                       _file_resources={'file-resource': os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg')})
        assert '鸡' in r or 'chicken' in r

        r = engine.run(gid, '你好，很高兴认识你')
        assert '你好' in r

        r = engine.run(gid, '生成音乐，长笛独奏，大自然之声。')
        assert '.wav' in r

    def test_stream_and_hostory(self):
        resources = [dict(id='0', kind='LocalLLM', name='base', args=dict(base_model='internlm2-chat-7b'))]
        builtin_history = [['水的沸点是多少？', '您好，我的答案是：水的沸点在标准大气压下是100摄氏度。'],
                           ['世界上最大的动物是什么？', '您好，我的答案是：蓝鲸是世界上最大的动物。'],
                           ['人一天需要喝多少水？', '您好，我的答案是：一般建议每天喝8杯水，大约2升。']]
        nodes = [dict(id='1', kind='SharedModel', name='m1', args=dict(llm='0', stream=True, prompt=dict(
                      system='请将我的问题翻译成中文。请注意，请直接输出翻译后的问题，不要反问和发挥',
                      user='问题: {query} \n, 翻译:'), cls='llm')),
                 dict(id='2', kind='SharedModel', name='m2',
                      args=dict(llm='0', stream=True,
                                prompt=dict(system='请参考历史对话，回答问题，并保持格式不变。', user='{query}'), cls='llm')),
                 dict(id='3', kind='JoinFormatter', name='join', args=dict(type='to_dict', names=['query', 'answer'])),
                 dict(id='4', kind='SharedModel', stream=False, name='m3',
                      args=dict(llm='0', history=builtin_history, cls='llm',
                                prompt=dict(system='你是一个问答机器人，会根据用户的问题作出回答。',
                                            user=('请结合历史对话和本轮的问题，总结我们的全部对话，无论是否相关。'
                                                  '本轮情况如下:\n {query}, 回答: {answer}'))))]
        engine = LightEngine()
        gid = engine.start(nodes, edges=[['__start__', '1'], ['1', '2'], ['1', '3'], ['2', '3'], ['3', '4'],
                                         ['4', '__end__']], resources=resources, _history_ids=['2', '4'])
        history = [['雨后为什么会有彩虹？', '您好，我的答案是：雨后阳光通过水滴发生折射和反射形成了彩虹。'],
                   ['月亮会发光吗？', '您好，我的答案是：月亮本身不会发光，它反射太阳光。'],
                   ['一年有多少天', '您好，我的答案是：一年有365天，闰年有366天。']]

        stream_result = ''
        with lazyllm.ThreadPoolExecutor(1) as executor:
            future = executor.submit(engine.run, gid, 'How many hours are there in a day?', _lazyllm_history=history)
            while True:
                if value := lazyllm.FileSystemQueue().dequeue():
                    stream_result += f"{''.join(value)}"
                elif future.done():
                    break
            result = future.result()
            assert '一天' in stream_result and '小时' in stream_result
            assert '您好，我的答案是' in stream_result and '24' in stream_result
            assert ('蓝鲸' in result or '动物' in result) and '水' in result

    def deploy_inference_service(self, model_name, deploy_method='auto', num_gpus=1):
        service_name = 'test_engine_infer_' + uuid.uuid4().hex

        data = {
            "service_name": service_name,
            "model_name": model_name,
            "framework": deploy_method,
            "num_gpus": num_gpus
        }
        response = requests.post(f"{self.infer_server_url}/v1/inference_services", json=data, headers=self.headers)
        assert response.status_code == 200

        for _ in range(30):  # wait 5 minutes
            response = requests.get(f"{self.infer_server_url}/v1/inference_services/{service_name}",
                                    headers=self.headers)
            assert response.status_code == 200
            response_data = response.json()
            if response_data['status'] == 'Ready':
                return model_name, response_data['deploy_method'], response_data['endpoint']
            elif response_data['status'] in ('Invalid', 'Cancelled', 'Failed'):
                raise RuntimeError(f'Deploy service failed. status is {response_data["status"]}')
            time.sleep(10)

        raise TimeoutError("inference service deploy timeout")

    def test_engine_infer_server(self):
        model_name = 'internlm2-chat-7b'
        model_name, deploy_method, url = self.deploy_inference_service(model_name)

        model = lazyllm.TrainableModule(model_name).deploy_method(getattr(lazyllm.deploy, deploy_method), url=url)
        assert model._impl._get_deploy_tasks.flag
        assert '你好' in model('请重复下面一句话：你好')

        engine = LightEngine()
        nodes = [dict(id='0', kind='LLM', name='m1',
                 args=dict(base_model=model_name, deploy_method=deploy_method, type='local', url=url, stream=True,
                           prompt=dict(system='请根据输入帮我计算，不要反问和发挥', user='输入: {query} \n, 答案:')))]
        gid = engine.start(nodes)
        r = engine.run(gid, '1 + 1 = ?')
        assert '2' in r

    def test_engine_infer_server_vqa(self):
        model_name = 'Mini-InternVL-Chat-2B-V1-5'
        model_name, deploy_method, url = self.deploy_inference_service(model_name, num_gpus=1)
        model = lazyllm.TrainableModule(model_name).deploy_method(getattr(lazyllm.deploy, deploy_method), url=url)
        assert model._impl._get_deploy_tasks.flag
        r = model("这张图片描述的是什么？", lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

        engine = LightEngine()
        nodes = [dict(id='0', kind='VQA', name='vqa',
                      args=dict(base_model=model_name, deploy_method=deploy_method, type='local', url=url))]
        gid = engine.start(nodes)

        r = engine.run(gid, "这张图片描述的是什么？", _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

    def test_engine_shared_vqa(self):
        engine = LightEngine()
        resources = [dict(id='vqa', kind='VQA', name='vqa',
                          args=dict(base_model='Mini-InternVL-Chat-2B-V1-5', type='local'))]

        nodes = [dict(id='0', kind='SharedModel', name='vqa',
                      args=dict(llm='vqa', cls='vqa'))]
        gid = engine.start(nodes, resources=resources)

        r = engine.run(gid, "这张图片描述的是什么？", _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

    def test_engine_infer_server_tts(self):
        model_name = 'ChatTTS-new'
        model_name, deploy_method, url = self.deploy_inference_service(model_name)
        model = lazyllm.TrainableModule(model_name).deploy_method(getattr(lazyllm.deploy, deploy_method), url=url)
        assert model._impl._get_deploy_tasks.flag
        assert '.wav' in model('你好啊，很高兴认识你。')

        engine = LightEngine()
        nodes = [dict(id='0', kind='TTS', name='tts',
                      args=dict(base_model=model_name, deploy_method=deploy_method, type='local', url=url))]
        gid = engine.start(nodes)

        r = engine.run(gid, "这张图片描述的是什么？")
        assert '.wav' in r

    def test_engine_tts_with_target_dir(self):
        engine = LightEngine()
        temp_dir = os.path.join(lazyllm.config['temp_dir'], 'tts_temp')
        nodes = [dict(id='1', kind='TTS', name='tts',
                      args=dict(base_model='ChatTTS-new', type='local', target_dir=temp_dir))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]
        gid = engine.start(nodes, edges)

        r = engine.run(gid, '测试TTS输出到指定目录')
        assert '.wav' in r
        assert os.path.exists(r)
        assert os.path.dirname(r) == temp_dir

    @pytest.mark.skip(reason='environment not ready')
    def test_OCR(self):
        nodes = [dict(id='1', kind='OCR', name='m1', args=dict(model="PP-OCRv5_mobile"))]
        edges = [dict(iid='__start__', oid='1'), dict(iid='1', oid='__end__')]
        data_root_dir = os.getenv("LAZYLLM_DATA_PATH")
        input = os.path.join(data_root_dir, "rag_master/default/__data/pdfs/reading_report_p1.pdf")
        engine = LightEngine()
        gid = engine.start(nodes, edges)
        data = engine.run(gid, input)
        verify = lazyllm.components.ocr.pp_ocr.OCR("PP-OCRv5_mobile")(input)
        assert len(data) == len(verify)

    def test_mcptool(self):
        resource = [dict(id='0', kind='LLM', name='base', args=dict(base_model='Qwen3-14B', type='local')),
                    dict(id='1', kind='MCPTool', name='list_allowed_directories',
                         args=dict(command_or_url="npx", tool_name="list_allowed_directories",
                                   args=["-y", "@modelcontextprotocol/server-filesystem", "./"]))]
        nodes = [dict(id='2', kind='FunctionCall', name='fc', args=dict(base_model='0', tools=['1']))]
        edges = [dict(iid='__start__', oid='2'), dict(iid='2', oid='__end__')]
        engine = LightEngine()
        gid = engine.start(nodes, edges, resources=resource)
        data = engine.run(gid, 'Show me your allowed directory.')
        assert isinstance(data, str)

        resource = [dict(id='3', kind='MCPTool', name='listdirectories',
                         args=dict(command_or_url="npx", tool_name="listdirectories",
                                   args=["-y", "@modelcontextprotocol/server-filesystem", "./"]))]
        nodes = [dict(id='4', kind='FunctionCall', name='fc', args=dict(base_model='0', tools=['3']))]
        edges = [dict(iid='__start__', oid='4'), dict(iid='4', oid='__end__')]
        with pytest.raises(AssertionError):
            gid = engine.start(nodes, edges, resources=resource)
