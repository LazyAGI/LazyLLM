import os
import json
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_fastapi_func

lazyllm.config.add("default_embedding_engine", str, "", "DEFAULT_EMBEDDING_ENGINE")

class Infinity(LazyLLMDeployBase):
    keys_name_handle = keys_name_handle = {
        'inputs': 'input',
    }
    message_format = {
        'input': 'who are you ?',
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self,
                 launcher=launchers.remote(ngpus=1),
                 **kw,
                 ):
        super().__init__(launcher=launcher)
        self.kw = ArgsDict({
            'host': '0.0.0.0',
            'port': None,
            'batch-size': 256,
        })
        self.kw.check_and_update(kw)
        self.random_port = False if 'port' in kw and kw['port'] else True

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
            cmd += self.kw.parse_kwargs()
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return 'http://{ip}:{port}/embeddings'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/embeddings'

    @staticmethod
    def extract_result(x, inputs):
        try:
            res_object = json.loads(x)
            res_list = [item['embedding'] for item in res_object['data']]
            if len(res_list) == 1 and type(inputs['input']) is str:
                res_list = res_list[0]
            return json.dumps(res_list)
        except Exception as e:
            LOG.warning(f'JSONDecodeError on load {x}')
            raise e
