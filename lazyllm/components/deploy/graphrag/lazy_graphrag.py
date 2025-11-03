# coding: utf-8
import json
import random
from pathlib import Path

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from ..base import LazyLLMDeployBase, verify_fastapi_func
from ..utils import get_log_path, make_log_dir


class GraphRAG(LazyLLMDeployBase):
    """GraphRAG deployment class that manages GraphRAG service as a subprocess"""

    keys_name_handle = {
        'query': 'query',
        'search_method': 'search_method',
        'community_level': 'community_level',
        'response_type': 'response_type',
    }

    message_format = {
        'query': 'What is the main topic?',
        'search_method': 'local',
        'community_level': 2,
        'response_type': 'Multiple Paragraphs',
    }

    default_headers = {'Content-Type': 'application/json'}
    target_name = 'query'

    def __init__(self, launcher=launchers.remote(ngpus=1), graphrag_executable=None, log_path=None, **kw):
        super().__init__(launcher=launcher)

        self.kw = ArgsDict({
            'host': '0.0.0.0',
            'port': None,
            'start_timeout': 60,
            'graphrag_executable': None,
        })
        self.options_keys = kw.pop('options_keys', [])
        self.kw.check_and_update(kw)

        if not graphrag_executable:
            raise ValueError('graphrag_executable must be provided')
        self.kw['graphrag_executable'] = graphrag_executable

        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'graphrag') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):
        kg_dir = base_model
        if not kg_dir or not Path(kg_dir).is_dir():
            raise ValueError("kg_dir must be provided and must be a directory")

        def impl():
            server_script = Path(__file__).parent / "graphrag_service_wrapper.py"

            if not server_script.exists():
                raise FileNotFoundError(f"GraphRAG server script not found: {server_script}")

            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            cmd_parts = [
                'python',
                str(server_script),
                '--graphrag_executable', self.kw['graphrag_executable'],
                '--kg_dir', str(kg_dir),
                '--host', self.kw['host'],
                '--port', str(self.kw['port']),
            ]

            if self.kw['start_timeout'] != 60:
                cmd_parts.extend(['--start_timeout', str(self.kw['start_timeout'])])

            cmd = ' '.join(cmd_parts)

            # Add logging if temp_folder is set
            if self.temp_folder:
                cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'

            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job

        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://<ip>:<port>/{self.target_name}'
        else:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/{self.target_name}'

    @staticmethod
    def extract_result(x, inputs):
        try:
            res_object = json.loads(x)
        except Exception as e:
            LOG.warning(f'JSONDecodeError on load {x}')
            raise e

        # GraphRAG service returns {"answer": "..."}
        if isinstance(res_object, dict) and 'answer' in res_object:
            return res_object['answer']
        else:
            LOG.warning(f'Unexpected response format: {res_object}')
            return ""

    def __repr__(self):
        return f"GraphRAG(host='{self.kw['host']}', port={self.kw['port']})"
