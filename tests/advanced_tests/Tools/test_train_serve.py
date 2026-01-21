import os
import time
import pytest
import requests

import lazyllm
from lazyllm.tools.train_service.serve import TrainServer
from urllib.parse import urlparse

class TestTrainServe:

    @pytest.mark.run_on_change('lazyllm/module/servermodule.py')
    def test_train_serve(self):
        train_server = lazyllm.ServerModule(TrainServer(), launcher=lazyllm.launcher.EmptyLauncher(sync=False))
        train_server.start()()
        parsed_url = urlparse(train_server._url)
        train_server_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        token = 'test_sft'
        headers = {'token': token}

        train_config = {
            'name': 'my_sft_model',
            'model': 'qwen1.5-0.5b-chat',
            'training_args': {
                'val_size': 0.1,
                'num_train_epochs': 100,
                'learning_rate': 0.1,
                'lr_scheduler_type': 'cosine',
                'per_device_train_batch_size': 32,
                'cutoff_len': 1024,
                'finetuning_type': 'lora',
                'lora_rank': 8,
                'lora_alpha': 32,
                'trust_remote_code': True,
                'ngpus': 1,
            },
            'training_dataset': [
                {
                    'dataset_download_uri': os.path.join(lazyllm.config['data_path'], 'alpaca/alpaca_data_zh_128.json'),
                    'format': 1,
                    'dataset_id': 'alpaca_zh_128'
                }
            ],
            'validation_dataset': [],
            'validate_dataset_split_percent': 0.1,
            'stage': 'SFT',
        }

        # Launch train
        response = requests.post(f'{train_server_url}/v1/finetuneTasks', json=train_config, headers=headers)
        job_id = response.json()['finetune_task_id']
        assert len(job_id) > 0
        status = response.json()['status']

        n = 0
        while status != 'Running':
            time.sleep(1)
            response = requests.get(f'{train_server_url}/v1/finetuneTasks/{job_id}', headers=headers)
            status = response.json()['status']
            n += 1
            assert n < 300, 'Launch training timeout.'

        # After Launch, training 20s
        time.sleep(20)

        response = requests.delete(f'{train_server_url}/v1/finetuneTasks/{job_id}', headers=headers)
        assert response.status_code == 200, response.text

        response = requests.get(f'{train_server_url}/v1/finetuneTasks/{job_id}', headers=headers)
        status = response.json()['status']
        cost = response.json()['cost']
        assert status == 'Cancelled'
        assert cost > 15

        response = requests.get(f'{train_server_url}/v1/finetuneTasks/{job_id}/log', headers=headers)
        assert response.status_code == 200, response.text

        response = requests.get(f'{train_server_url}/v1/finetuneTasks/jobs', headers=headers)
        assert response.status_code == 200, response.text
        assert job_id in response.json().keys()

        train_server.stop()
