import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func
from .utils import get_log_path, make_log_dir

lazyllm.config.add("default_embedding_engine", str, "", "DEFAULT_EMBEDDING_ENGINE")

class Infinity(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'input',
    }
    message_format = {
        'input': 'who are you ?',
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=launchers.remote(ngpus=1), model_type='embed', log_path=None, **kw):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'host': '0.0.0.0',
            'port': None,
            'batch-size': 256,
        })
        self._model_type = model_type
        kw.pop('stream', '')
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] else True
        if self._model_type == "reranker":
            self._update_reranker_message()
        self.temp_folder = make_log_dir(log_path, 'lmdeploy') if log_path else None

    def _update_reranker_message(self):
        self.keys_name_handle = {
            'inputs': 'query',
        }
        self.message_format = {
            'query': 'who are you ?',
            'documents': ['string'],
            'return_documents': False,
            'raw_scores': False,
            'top_n': 1,
            'model': 'default/not-specified',
        }
        self.default_headers = {'Content-Type': 'application/json'}

    def cmd(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model

        def impl():
            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)
            cmd = f'infinity_emb v2 --model-id {finetuned_model} '
            if isinstance(self._launcher, launchers.EmptyLauncher) and self._launcher.ngpus:
                available_gpus = self._launcher._get_idle_gpus()
                required_count = self._launcher.ngpus
                if required_count <= len(available_gpus):
                    gpu_ids = ','.join(map(str, available_gpus[:required_count]))
                    cmd += f'--device-id={gpu_ids} '
                else:
                    raise RuntimeError(
                        f"Insufficient GPUs available (required: {required_count}, "
                        f"available: {len(available_gpus)})"
                    )
            cmd += self.kw.parse_kwargs()
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if self._model_type == "reranker":
            target_name = 'rerank'
        else:
            target_name = 'embeddings'
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://<ip>:<port>/{target_name}'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/{target_name}'

    @staticmethod
    def extract_result(x, inputs):
        try:
            res_object = json.loads(x)
        except Exception as e:
            LOG.warning(f'JSONDecodeError on load {x}')
            raise e
        assert 'object' in res_object
        object_type = res_object['object']
        if object_type == 'list':  # for infinity >= 0.0.64
            object_type = res_object['data'][0]['object']
        if object_type == 'embedding':
            res_list = [item['embedding'] for item in res_object['data']]
            if len(res_list) == 1 and type(inputs['input']) is str:
                res_list = res_list[0]
            return json.dumps(res_list)
        elif object_type == 'rerank':
            return [(x['index'], x['relevance_score']) for x in res_object['results']]
