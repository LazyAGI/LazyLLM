import os
import json
import requests
from urllib.parse import urljoin

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.utils import update_config, TrainConfig, uniform_sft_dataset


class LocalTrainClient:

    def __init__(self, url):
        self.url = url

    def uniform_status(self, status):
        res = 'Invalid'
        if Status[status] == Status.Done:
            res = 'Done'
        elif Status[status] == Status.Cancelled:
            res = 'Cancelled'
        elif Status[status] == Status.Failed:
            res = 'Failed'
        elif Status[status] == Status.Running:
            res = 'Running'
        else:  # TBSubmitted, InQueue, Pending
            res = 'Pending'
        return res

    def register_token(self, token):
        url = urljoin(self.url, f'register_bearer?bearer={token}')
        try:
            response = requests.post(url)
            response.raise_for_status()
        except Exception:
            pass

    def train(self, train_config, token='default'):
        self.register_token(token)  # register token as user_group

        url = urljoin(self.url, 'v1/fine_tuning/jobs')
        headers = {
            "Content-Type": "application/json",
            "token": token,
        }
        train_config = update_config(train_config, TrainConfig)
        data = {
            'finetune_model_name': train_config['finetune_model_name'],
            'base_model': train_config['base_model'],
            'data_path': train_config['data_path'],
            'hyperparameters': {
                'stage': train_config['training_type'].strip().lower(),
                'finetuning_type': train_config['finetuning_type'].strip().lower(),
                'val_size': train_config['val_size'],
                'num_train_epochs': train_config['num_epochs'],
                'learning_rate': train_config['learning_rate'],
                'lr_scheduler_type': train_config['lr_scheduler_type'],
                'per_device_train_batch_size': train_config['batch_size'],
                'cutoff_len': train_config['cutoff_len'],
                'lora_r': train_config['lora_r'],
                'lora_alpha': train_config['lora_alpha'],
            }
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        res = response.json()
        if 'status' in res:
            status = self.uniform_status(res['status'])
            job_id = res['job_id']
            return (job_id, status)
        else:
            return res

    def cancel_finetuning(self, token, job_id):
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}/cancel')
        headers = {
            "token": token,
        }
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            status = response.json()['status']
        except Exception as e:
            status = str(e)
            lazyllm.LOG.error(status)
        if status == 'Cancelled':
            return "Successfully cancelled task."
        else:
            return "Failed to cancel task. " + (f" Because: {status}" if status else '')

    def get_train_status(self, token, job_id):
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            status = self.uniform_status(response.json()['status'])
        except Exception as e:
            status = 'Invalid'
            lazyllm.LOG.error(e)
        return status

    def get_target_model():
        pass

    def get_log(self, token, job_id):
        url = urljoin(self.url, f'v1/fine_tuning/jobs/{job_id}/events')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            lazyllm.LOG.error(f"Failed to get log. Because: {e}")
            return None
        return response.json()['log']

    def get_all_finetuned_models(self, token):
        url = urljoin(self.url, 'v1/fine_tuning/jobs')
        headers = {"token": token}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            lazyllm.LOG.error(f"Failed to get log. Because: {e}")
            return [], []
        model_data = response.json()
        names_valid = []
        names_invalid = []
        for job_id, job in model_data.items():
            if job['status'] == 'Done':
                names_valid.append((job_id, job['fine_tuned_model']))
            else:
                names_invalid.append((job_id, job['fine_tuned_model']))
        return names_valid, names_invalid

class OnlineTrainClient:

    def __init__(self):
        pass

    def train(self, train_config, token, source='glm'):
        train_config = update_config(train_config, TrainConfig)
        assert train_config['training_type'].lower() == 'sft', 'Only supported sft!'

        data_path = os.path.join(lazyllm.config['data_path'], train_config['data_path'])
        data_path = uniform_sft_dataset(data_path, target='openai')
        m = lazyllm.OnlineChatModule(model=train_config['base_model'], api_key=token, source=source)

        file_id = m._upload_train_file(train_file=data_path)
        fine_tuning_job_id, status = m._create_finetuning_job(m._model_name, file_id, **train_config)

        return fine_tuning_job_id, status

    def get_all_finetuned_models(self, token, source='glm'):
        m = lazyllm.OnlineChatModule(source=source, api_key=token)
        return m._get_finetuned_model_names()

    def get_train_status(self, token, job_id, source='glm'):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            status = m._query_job_status(job_id)
        except Exception as e:
            status = 'Invalid'
            lazyllm.LOG.error(e)
        return status

    def cancel_finetuning(self, token, job_id, source='glm'):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            res = m._cancel_finetuning_job(job_id)
        except Exception as e:
            res = str(e)
        if res == 'Cancelled':
            return "Successfully cancelled task."
        else:
            return "Failed to cancel task. " + (f" Because: {res}" if res else '')

    def get_log(self, token, job_id, source='glm', target_path=None):
        try:
            m = lazyllm.OnlineChatModule(source=source, api_key=token)
            file_name, log = m._get_log(job_id)
        except Exception as e:
            lazyllm.LOG.error(f"Failed to get log. Because: {e}")
            return None
        save_path = target_path if target_path else os.path.join(m._get_temp_save_dir_path(), f'{file_name}.log')
        with open(save_path, 'w', encoding='utf-8') as log_file:
            json.dump(log, log_file, indent=4, ensure_ascii=False)
        return save_path
