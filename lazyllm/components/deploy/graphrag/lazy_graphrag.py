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

    keys_name_handle = {'inputs': 'query'}

    message_format = {
        'query': 'What is the main topic?',
        'search_method': 'local',
        'community_level': 2,
        'response_type': 'Multiple Paragraphs',
    }

    default_headers = {'Content-Type': 'application/json'}
    target_name = 'query'

    def __init__(self, launcher=launchers.remote(ngpus=1), graphrag_executable=None, kg_dir=None, log_path=None, **kw):
        super().__init__(launcher=launcher)

        self.kw = ArgsDict({
            'host': '0.0.0.0',
            'port': None,
            'start_timeout': 60,
            'graphrag_executable': None,
            'kg_dir': None,
        })
        self.options_keys = kw.pop('options_keys', [])
        self.kw.check_and_update(kw)

        if not graphrag_executable:
            raise ValueError('graphrag_executable must be provided')
        self.kw['graphrag_executable'] = graphrag_executable

        if not kg_dir:
            raise ValueError('kg_dir must be provided')
        self.kw['kg_dir'] = kg_dir

        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'graphrag') if log_path else None

    def cmd(self, finetuned_model=None, base_model=None):

        def impl_old():
            server_script = Path(__file__).parent / "graphrag_service_wrapper.py"

            if not server_script.exists():
                raise FileNotFoundError(f"GraphRAG server script not found: {server_script}")

            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            cmd_parts = [
                'python',
                str(server_script),
                '--graphrag_executable', self.kw['graphrag_executable'],
                '--kg_dir', self.kw['kg_dir'],
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

        def impl():
            python_executable = (Path(self.kw['graphrag_executable']) / '../python').resolve()
            current_dir = Path(__file__).parent
            service_script = current_dir / 'graphrag_service.py'

            if not service_script.exists():
                raise FileNotFoundError(f"GraphRAG server script not found: {service_script}")

            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            cmd_parts = [
                str(python_executable),
                str(service_script),
                '--kg_dir', self.kw['kg_dir'],
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
