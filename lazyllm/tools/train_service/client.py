import os
import json
import requests
from urllib.parse import urljoin

import lazyllm
from lazyllm.module.llms.utils import update_config, TrainConfig, uniform_sft_dataset
from ..services import ClientBase


class LocalTrainClient(ClientBase):

    def __init__(self, url):
        super().__init__(urljoin(url, 'v1/fine_tuning/'))

    def train(self, train_config, token):
        url = urljoin(self.url, 'jobs')
        headers = {
            'Content-Type': 'application/json',
            'token': token,
        }
        train_config = update_config(train_config, TrainConfig)
        data = {
            'finetune_model_name': train_config['finetune_model_name'],
            'base_model': train_config['base_model'],
            'data_path': train_config['data_path'],
            'num_gpus': train_config['num_gpus'],
            'hyperparameters': {
                'stage': train_config['training_type'].strip().lower(),
                'finetuning_type': train_config['finetuning_type'].strip().lower(),
                'val_size': train_config['val_size'],
                'num_train_epochs': train_config['num_epochs'],
                'learning_rate': train_config['learning_rate'],
                'lr_scheduler_type': train_config['lr_scheduler_type'],
                'per_device_train_batch_size': train_config['batch_size'] // train_config['num_gpus'],
                'cutoff_len': train_config['cutoff_len'],
                'lora_r': train_config['lora_r'],
                'lora_alpha': train_config['lora_alpha'],
            }
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res = response.json()
            return (res['job_id'], self.uniform_status(res['status']))
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return (None, str(e))

    def cancel_training(self, token, job_id):
        url = urljoin(self.url, f'jobs/{job_id}/cancel')
        headers = {
            'token': token,
        }
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            status = response.json()['status']
            if status == 'Cancelled':
                return True
            else:
                return f'Failed to cancel task. Final status is {status}'
        except Exception as e:
            status = str(e)
            lazyllm.LOG.error(str(e))
            return f'Failed to cancel task. Because: {str(e)}'

    def get_training_cost(self, token, job_id):
        url = urljoin(self.url, f'jobs/{job_id}')
        headers = {'token': token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()['cost']
        except Exception as e:
            error = f'Failed to get cost. Because: {str(e)}'
            lazyllm.LOG.error(error)
            return error

    def get_training_status(self, token, job_id):
        url = urljoin(self.url, f'jobs/{job_id}')
        headers = {'token': token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = self.uniform_status(response.json()['status'])
        except Exception as e:
            status = 'Invalid'
            lazyllm.LOG.error(str(e))
        return status

    def get_training_log(self, token, job_id):
        url = urljoin(self.url, f'jobs/{job_id}/events')
        headers = {'token': token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()['log']
        except Exception as e:
            lazyllm.LOG.error(f'Failed to get log. Because: {str(e)}')
            return None

    def get_all_trained_models(self, token):
        url = urljoin(self.url, 'jobs')
        headers = {'token': token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            model_data = response.json()
            res = list()
            for job_id, job in model_data.items():
                res.append([job_id, job['fine_tuned_model'], job['status']])
            return res
        except Exception as e:
            lazyllm.LOG.error(f'Failed to get log. Because: {e}')
            return None

class OnlineTrainClient:

    def __init__(self):
        pass

    def train(self, train_config, token, source):
        try:
            train_config = update_config(train_config, TrainConfig)
            assert train_config['training_type'].lower() == 'sft', 'Only supported sft!'

            data_path = os.path.join(lazyllm.config['data_path'], train_config['data_path'])
            data_path = uniform_sft_dataset(data_path, target='openai')
            m = lazyllm.OnlineChatModule(model=train_config['base_model'], api_key=token, source=source)

            file_id = m._upload_train_file(train_file=data_path)
            fine_tuning_job_id, status = m._create_finetuning_job(m._model_name, file_id, **train_config)

            return (fine_tuning_job_id, status)
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return (None, str(e))

    def get_all_trained_models(self, token, source):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            return m._get_finetuned_model_names()
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return None

    def get_training_status(self, token, job_id, source):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            status = m._query_job_status(job_id)
        except Exception as e:
            status = 'Invalid'
            lazyllm.LOG.error(e)
        return status

    def cancel_training(self, token, job_id, source):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            res = m._cancel_finetuning_job(job_id)
            if res == 'Cancelled':
                return True
            else:
                return f'Failed to cancel task. Final info is {res}'
        except Exception as e:
            lazyllm.LOG.error(str(e))
            return f'Failed to cancel task. Because: {str(e)}'

    def get_training_log(self, token, job_id, source, target_path=None):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            file_name, log = m._get_log(job_id)
            save_path = target_path if target_path else os.path.join(m._get_temp_save_dir_path(), f'{file_name}.log')
            with open(save_path, 'w', encoding='utf-8') as log_file:
                json.dump(log, log_file, indent=4, ensure_ascii=False)
            return save_path
        except Exception as e:
            lazyllm.LOG.error(f'Failed to get log. Because: {e}')
            return None

    def get_training_cost(self, token, job_id, source):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            res = m._query_finetuning_cost(job_id)
            return res
        except Exception as e:
            error = f'Failed to get cost. Because: {str(e)}'
            lazyllm.LOG.error(error)
            return error

    def validate_api_key(self, token, source, secret_key=None):
        m = lazyllm.OnlineChatModule(source=source, api_key=token, secret_key=secret_key)
        return m._validate_api_key()
