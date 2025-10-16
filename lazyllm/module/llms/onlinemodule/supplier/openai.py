import json
import os
import uuid
import requests
from typing import Tuple, List, Dict, Union
from urllib.parse import urljoin
import lazyllm
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase
from ..fileHandler import FileHandlerBase

class OpenAIModule(OnlineChatModuleBase, FileHandlerBase):
    TRAINABLE_MODEL_LIST = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106',
                            'gpt-3.5-turbo-0613', 'babbage-002',
                            'davinci-002', 'gpt-4-0613']
    NO_PROXY = False

    def __init__(self, base_url: str = 'https://api.openai.com/v1/', model: str = 'gpt-3.5-turbo',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, skip_auth: bool = False, **kw):
        OnlineChatModuleBase.__init__(self, model_series='OPENAI', api_key=api_key or lazyllm.config['openai_api_key'],
                                      base_url=base_url, model_name=model, stream=stream, return_trace=return_trace,
                                      skip_auth=skip_auth, **kw)
        FileHandlerBase.__init__(self)
        self.default_train_data = {
            'model': 'gpt-3.5-turbo-0613',
            'training_file': None,
            'validation_file': None,
            'hyperparameters': {
                'n_epochs': 1,
                'batch_size': 16,
                'learning_rate_multiplier': '1.6e-5',
            }
        }
        self.fine_tuning_job_id = None

    def _get_system_prompt(self):
        return 'You are ChatGPT, a large language model trained by OpenAI.You are a helpful assistant.'

    def _convert_file_format(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as fr:
            dataset = [json.loads(line) for line in fr]

        json_strs = []
        for ex in dataset:
            lineEx = {'messages': []}
            messages = ex.get('messages', [])
            for message in messages:
                role = message.get('role', '')
                content = message.get('content', '')
                if role in ['system', 'user', 'assistant']:
                    lineEx['messages'].append({'role': role, 'content': content})
            json_strs.append(json.dumps(lineEx, ensure_ascii=False))

        return '\n'.join(json_strs)

    def _upload_train_file(self, train_file):
        headers = {
            'Authorization': 'Bearer ' + self._api_key
        }

        url = urljoin(self._base_url, 'files')

        self.get_finetune_data(train_file)

        file_object = {
            'purpose': (None, 'fine-tune', None),
            'file': (os.path.basename(train_file), self._dataHandler, 'application/json')
        }

        with requests.post(url, headers=headers, files=file_object) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            # delete temporary training file
            self._dataHandler.close()
            return r.json()['id']

    def _update_kw(self, data, normal_config):
        current_train_data = self.default_train_data.copy()
        current_train_data.update(data)

        current_train_data['hyperparameters']['n_epochs'] = normal_config['num_epochs']
        current_train_data['hyperparameters']['learning_rate_multiplier'] = str(normal_config['learning_rate'])
        current_train_data['hyperparameters']['batch_size'] = normal_config['batch_size']
        current_train_data['suffix'] = str(uuid.uuid4())[:7]

        return current_train_data

    def _create_finetuning_job(self, train_model, train_file_id, **kw) -> Tuple[str, str]:
        url = urljoin(self._base_url, 'fine_tuning/jobs')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }
        data = {
            'model': train_model,
            'training_file': train_file_id
        }
        if len(kw) > 0:
            if 'finetuning_type' in kw:
                data = self._update_kw(data, kw)
            else:
                data.update(kw)

        with requests.post(url, headers=headers, json=data) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            fine_tuning_job_id = r.json()['id']
            self.fine_tuning_job_id = fine_tuning_job_id
            status = r.json()['status']
            return (fine_tuning_job_id, status)

    def _cancel_finetuning_job(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            return 'Invalid'
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = urljoin(self._base_url, f'fine_tuning/jobs/{job_id}/cancel')
        headers = {
            'Authorization': f'Bearer {self._api_key}'
        }
        with requests.post(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        status = r.json()['status']
        if status == 'cancelled':
            return 'Cancelled'
        else:
            return f'JOB {job_id} status: {status}'

    def _query_finetuned_jobs(self):
        fine_tune_url = urljoin(self._base_url, 'fine_tuning/jobs')
        headers = {
            'Authorization': f'Bearer {self._api_key}',
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _get_finetuned_model_names(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        model_data = self._query_finetuned_jobs()
        res = list()
        for model in model_data['data']:
            res.append([model['id'], model['fine_tuned_model'], self._status_mapping(model['status'])])
        return res

    def _status_mapping(self, status):
        if status == 'succeeded':
            return 'Done'
        elif status == 'failed':
            return 'Failed'
        elif status == 'cancelled':
            return 'Cancelled'
        elif status == 'running':
            return 'Running'
        else:  # validating_files, queued
            return 'Pending'

    def _query_job_status(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError('No job ID specified. Please ensure that a valid "fine_tuning_job_id" is '
                               'provided as an argument or started a training job.')
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        _, status = self._query_finetuning_job(job_id)
        return self._status_mapping(status)

    def _get_log(self, fine_tuning_job_id=None):
        if not fine_tuning_job_id and not self.fine_tuning_job_id:
            raise RuntimeError('No job ID specified. Please ensure that a valid "fine_tuning_job_id" is '
                               'provided as an argument or started a training job.')
        job_id = fine_tuning_job_id if fine_tuning_job_id else self.fine_tuning_job_id
        fine_tune_url = urljoin(self._base_url, f'fine_tuning/jobs/{job_id}/events')
        headers = {
            'Authorization': f'Bearer {self._api_key}'
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return job_id, r.json()

    def _get_curr_job_model_id(self):
        if not self.fine_tuning_job_id:
            return None, None
        model_id, _ = self._query_finetuning_job(self.fine_tuning_job_id)
        return self.fine_tuning_job_id, model_id

    def _query_finetuning_job_info(self, fine_tuning_job_id):
        fine_tune_url = urljoin(self._base_url, f'fine_tuning/jobs/{fine_tuning_job_id}')
        headers = {
            'Authorization': f'Bearer {self._api_key}'
        }
        with requests.get(fine_tune_url, headers=headers) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        return r.json()

    def _query_finetuning_job(self, fine_tuning_job_id) -> Tuple[str, str]:
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        status = info['status']
        fine_tuned_model = info['fine_tuned_model'] if 'fine_tuned_model' in info else None
        return (fine_tuned_model, status)

    def _query_finetuning_cost(self, fine_tuning_job_id):
        info = self._query_finetuning_job_info(fine_tuning_job_id)
        if 'trained_tokens' in info and info['trained_tokens']:
            return info['trained_tokens']
        else:
            return None

    def _create_deployment(self) -> Tuple[str, str]:
        return (self._model_name, 'RUNNING')

    def _query_deployment(self, deployment_id) -> str:
        return 'RUNNING'


class OpenAIEmbedding(OnlineEmbeddingModuleBase):
    NO_PROXY = True

    def __init__(self,
                 embed_url: str = 'https://api.openai.com/v1/',
                 embed_model_name: str = 'text-embedding-ada-002',
                 api_key: str = None, batch_size: int = 16, **kw):
        super().__init__('OPENAI', embed_url, api_key or lazyllm.config['openai_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)

    def _set_embed_url(self):
        self._embed_url = urljoin(self._embed_url, 'embeddings')


class OpenAIReranking(OnlineEmbeddingModuleBase):
    NO_PROXY = True

    def __init__(self,
                 embed_url: str = 'https://api.openai.com/v1/',
                 embed_model_name: str = '',
                 api_key: str = None,
                 **kw):
        super().__init__('OPENAI', embed_url, api_key or lazyllm.config['openai_api_key'], embed_model_name)

    def _set_embed_url(self):
        self._embed_url = urljoin(self._embed_url, 'rerank')

    @property
    def type(self):
        return 'RERANK'

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            'query': query,
            'documents': documents,
            'top_n': top_n,
            'model': self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        results = response['results']
        return [(result['index'], result['relevance_score']) for result in results]
