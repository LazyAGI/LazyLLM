import os
import random

from lazyllm import launchers, LazyLLMCMD, bind, _0, ArgsDict
from .base import LazyLLMDeployBase, restart_service


class Lightllm(LazyLLMDeployBase):
    input_key_name = 'inputs'
    default_headers = {'Content-Type': 'application/json'}
    message_formate ={
        input_key_name: 'Who are you ?',
        'parameters': {
            'do_sample': False,
            'ignore_eos': False,
            'max_new_tokens': 512,
            'temperature': 0.1,
        }
    }

    def __init__(self,
                 trust_remote_code=True,
                 llm_launcher=launchers.slurm,
                 **kw,
                 ):
        super().__init__(launcher=llm_launcher)
        self.kw = ArgsDict({
            'tp': 1,
            'max_total_token_num': 64000,
            'eos_id': 2,
            'port': None,
            'host':'0.0.0.0',
            'nccl_port': None,
            'tokenizer_mode': 'auto',
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        

    def cmd(self, model_dir=None, base_model=None):
        if not os.path.exists(model_dir) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(model_dir)):
            if not model_dir:
                print(f"Note! That model_dir({model_dir}) is an invalid path, "
                    f"base_model({base_model}) will be used")
            model_dir = base_model
        def build_cmd():
            if not self.kw['port']:
                self.kw['port'] = random.randint(30000, 40000)
            if not self.kw['nccl_port']:
                self.kw['nccl_port'] = random.randint(20000, 30000)

            cmd = (
                'python -m lightllm.server.api_server '
                f'--model_dir {model_dir} '
                )
            cmd += self.kw.parse_kwargs()
            if self.trust_remote_code:
                cmd += ' --trust_remote_code '
            return cmd, self.kw['port']
        func = build_cmd if not self.kw['port'] else None
        cmd, port = build_cmd()
        return LazyLLMCMD(cmd=cmd,
                          post_function=bind(
                              restart_service,
                              _0,
                              port,
                              self.default_headers,
                              self.message_formate,
                              func)
                        )
