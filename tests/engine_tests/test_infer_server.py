import uuid
import os
import time
import requests
import lazyllm
from lazyllm.engine import LightEngine
from lazyllm.launcher import cleanup
from lazyllm.tools.infer_service.serve import InferServer
from urllib.parse import urlparse
import pytest


class TestInferServer:
    def setup_method(self):
        self.infer_server = lazyllm.ServerModule(InferServer(), launcher=lazyllm.launcher.EmptyLauncher(sync=False))
        self.infer_server.start()()
        parsed_url = urlparse(self.infer_server._url)
        self.infer_server_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        token = '123'
        self.headers = {'token': token}

    def teardown_method(self):
        self.infer_server.stop()
        cleanup()

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        LightEngine().reset()
        lazyllm.FileSystemQueue().dequeue()
        lazyllm.FileSystemQueue(klass='lazy_trace').dequeue()

    def deploy_inference_service(self, model_name, deploy_method='auto', num_gpus=1):
        service_name = 'test_engine_infer_' + uuid.uuid4().hex

        data = {
            'service_name': service_name,
            'model_name': model_name,
            'framework': deploy_method,
            'num_gpus': num_gpus
        }
        response = requests.post(f'{self.infer_server_url}/v1/inference_services', json=data, headers=self.headers)
        assert response.status_code == 200

        for _ in range(30):  # wait 5 minutes
            response = requests.get(f'{self.infer_server_url}/v1/inference_services/{service_name}',
                                    headers=self.headers)
            assert response.status_code == 200
            response_data = response.json()
            if response_data['status'] == 'Ready':
                return model_name, response_data['deploy_method'], response_data['endpoint'], service_name
            elif response_data['status'] in ('Invalid', 'Cancelled', 'Failed'):
                raise RuntimeError(f'Deploy service failed. status is {response_data["status"]}')
            time.sleep(10)

        raise TimeoutError('inference service deploy timeout')

    def delete_inference_service(self, service_name):
        response = requests.delete(f'{self.infer_server_url}/v1/inference_services/{service_name}', headers=self.headers)
        assert response.status_code == 200

    @pytest.mark.run_on_change(
        'lazyllm/tools/infer_service/serve.py',
        'lazyllm/tools/services/services.py',
        'lazyllm/engine/lightengine.py')
    def test_engine_infer_server(self):
        model_name = 'internlm2-chat-7b'
        model_name, deploy_method, url, service_name = self.deploy_inference_service(model_name)

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

        self.delete_inference_service(service_name)

    @pytest.mark.run_on_change(
        'lazyllm/tools/infer_service/serve.py',
        'lazyllm/tools/services/services.py',
        'lazyllm/engine/lightengine.py')
    def test_engine_infer_server_vqa(self):
        model_name = 'InternVL3_5-1B'
        model_name, deploy_method, url, service_name = self.deploy_inference_service(
            model_name, deploy_method='lmdeploy', num_gpus=1)
        model = lazyllm.TrainableModule(model_name).deploy_method(getattr(lazyllm.deploy, deploy_method), url=url)
        assert model._impl._get_deploy_tasks.flag
        r = model('这张图片描述的是什么？', lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

        engine = LightEngine()
        nodes = [dict(id='0', kind='VQA', name='vqa',
                      args=dict(base_model=model_name, deploy_method=deploy_method, type='local', url=url))]
        gid = engine.start(nodes)

        r = engine.run(gid, '这张图片描述的是什么？', _lazyllm_files=os.path.join(lazyllm.config['data_path'], 'ci_data/ji.jpg'))
        assert '鸡' in r or 'chicken' in r

        self.delete_inference_service(service_name)

    @pytest.mark.run_on_change(
        'lazyllm/tools/infer_service/serve.py',
        'lazyllm/tools/services/services.py',
        'lazyllm/engine/lightengine.py')
    def test_engine_infer_server_tts(self):
        model_name = 'bark'
        model_name, deploy_method, url, service_name = self.deploy_inference_service(model_name)
        model = lazyllm.TrainableModule(model_name).deploy_method(getattr(lazyllm.deploy, deploy_method), url=url)
        assert model._impl._get_deploy_tasks.flag
        assert '.wav' in model('你好啊，很高兴认识你。')

        engine = LightEngine()
        nodes = [dict(id='0', kind='TTS', name='tts',
                      args=dict(base_model=model_name, deploy_method=deploy_method, type='local', url=url))]
        gid = engine.start(nodes)

        r = engine.run(gid, '这张图片描述的是什么？')
        assert '.wav' in r

        self.delete_inference_service(service_name)
