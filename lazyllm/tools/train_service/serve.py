import os
import time
import uuid
import string
import random
import asyncio
import threading
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException, Header
from async_timeout import timeout

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.utils import uniform_sft_dataset
from lazyllm import FastapiApp as app
from ..services import ServerBase


class JobDescription(BaseModel):
    finetune_model_name: str
    base_model: str = Field(default='qwen1.5-0.5b-chat')
    data_path: str = Field(default='alpaca/alpaca_data_zh_128.json')
    num_gpus: int = Field(default=1)
    hyperparameters: dict = Field(
        default={
            'stage': 'sft',
            'finetuning_type': 'lora',
            'val_size': 1,
            'num_train_epochs': 1,
            'learning_rate': 0.0001,
            'lr_scheduler_type': 'cosine',
            'per_device_train_batch_size': 16,
            'cutoff_len': 1024,
            'lora_r': 8,
            'lora_alpha': 32,
        }
    )

class TrainServer(ServerBase):

    def _update_status(self, token, job_id):
        if not self._in_active_jobs(token, job_id):
            return
        # Get basic info
        info = self._read_user_job_info(token, job_id)
        save_path = info['fine_tuned_model']
        log_path = info['log_path']

        # Get status
        m, _ = self._read_active_job(token, job_id)
        status = m.status(info['model_id']).name

        update = {'status': status}

        # Some tasks not run when they are just created
        if Status[status] == Status.Running and not info['started_at']:
            update = {
                'status': status,
                'started_at': datetime.now().strftime(self._time_format),
            }

        # Some tasks cannot obtain the storage path when they are just started
        if not save_path:
            update['fine_tuned_model'] = self._get_save_path(m)
        if not log_path:
            update['log_path'] = self._get_log_path(m)

        # Update Status
        self._update_user_job_info(token, job_id, update)

        # Pop and kill jobs with status: Done, Failed
        if Status[status] in (Status.Done, Status.Failed):
            m, _ = self._pop_active_job(token, job_id)
            m.stop(info['model_id'])
            if info['started_at'] and not info['cost']:
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_info(token, job_id, {'cost': cost})
            return

        create_time = datetime.strptime(info['created_at'], self._time_format)
        delta_time = (datetime.now() - create_time).total_seconds()

        # More than 5 min pop and kill jobs with status: Cancelled. Because of
        # some tasks have just been started and their status cannot be checked.
        if delta_time > 300 and Status[status] == Status.Cancelled:
            m, _ = self._pop_active_job(token, job_id)
            m.stop(info['model_id'])
            if info['started_at'] and not info['cost']:
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_info(token, job_id, {'cost': cost})
            return

        # More than 50 min pop and kill jobs with status: TBSubmitted, InQueue, Pending
        if delta_time > 3000 and Status[status] in (Status.TBSubmitted, Status.InQueue, Status.Pending):
            m, _ = self._pop_active_job(token, job_id)
            m.stop(info['model_id'])
            return

    def _get_save_path(self, model):
        if not hasattr(model._impl, '_temp_finetuned_model_path'):
            return None
        return model._impl._temp_finetuned_model_path

    def _get_log_path(self, model):
        log_dir = self._get_save_path(model)
        if not log_dir:
            return None

        parts = log_dir.split(os.sep)
        if parts[-1].endswith('lazyllm_merge'):
            parts[-1] = parts[-1].replace('lazyllm_merge', 'lazyllm_lora')
        log_dir = os.sep.join(parts)

        log_files_paths = []
        for file in os.listdir(log_dir):
            if file.endswith('.log') and file.startswith('train_log_'):
                log_files_paths.append(os.path.join(log_dir, file))
        if len(log_files_paths) == 0:
            return None
        assert len(log_files_paths) == 1
        return log_files_paths[-1]

    @app.post('/v1/fine_tuning/jobs')
    async def create_job(self, job: JobDescription, token: str = Header(None)):
        # await self.authorize_current_user(token)
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        # Build Job-ID:
        create_time = datetime.now().strftime(self._time_format)
        job_id = '-'.join(['ft', create_time, str(uuid.uuid4())[:5]])

        # Build Model-ID:
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=7))
        model_id = job.finetune_model_name + '_' + random_string

        # Build checkpoint save dir:
        # - No-Env-Set: (work/path + save_ckpt) + token + job_id;
        # - Env-Set:    (train_target_root)     + token + job_id;
        save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)

        # Add launcher into hyperparameters:
        hypram = job.hyperparameters
        hypram['ngpus'] = job.num_gpus

        # Uniform Training DataSet:
        job.data_path = os.path.join(lazyllm.config['data_path'], job.data_path)
        job.data_path = uniform_sft_dataset(job.data_path, target='alpaca')

        # Set params for TrainableModule:
        m = lazyllm.TrainableModule(job.base_model, save_root)\
            .trainset(job.data_path)\
            .finetune_method(lazyllm.finetune.llamafactory)

        # Launch Training:
        thread = threading.Thread(target=m._impl._async_finetune, args=(model_id,), kwargs=hypram)
        thread.start()

        # Sleep 5s for launch cmd.
        try:
            async with timeout(5):
                while m.status(model_id) == Status.Cancelled:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            pass

        # The first getting the path may be invalid, and it will be getted with each update.
        save_path = self._get_save_path(m)
        log_path = self._get_log_path(m)

        # Save status
        status = m.status(model_id).name
        if Status[status] == Status.Running:
            started_time = datetime.now().strftime(self._time_format)
        else:
            started_time = None
        self._update_active_jobs(token, job_id, (m, thread))
        self._update_user_job_info(token, job_id, {
            'model_id': model_id,
            'job_id': job_id,
            'base_model': job.base_model,
            'created_at': create_time,
            'fine_tuned_model': save_path,
            'status': status,
            'data_path': job.data_path,
            'hyperparameters': hypram,
            'log_path': log_path,
            'started_at': started_time,
            'cost': None,
        })

        return {'job_id': job_id, 'status': status}

    @app.post('/v1/fine_tuning/jobs/{job_id}/cancel')
    async def cancel_job(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_active_jobs(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        m, _ = self._pop_active_job(token, job_id)
        info = self._read_user_job_info(token, job_id)
        m.stop(info['model_id'])

        total_sleep = 0
        while m.status(info['model_id']) != Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 10:
                raise HTTPException(status_code=404, detail=f'Task {job_id}, ccancelled timed out.')

        status = m.status(info['model_id']).name
        update_dict = {'status': status}
        if info['started_at'] and not info['cost']:
            update_dict['cost'] = (datetime.now() - datetime.strptime(info['started_at'],
                                                                      self._time_format)).total_seconds()
        self._update_user_job_info(token, job_id, update_dict)

        return {'status': status}

    @app.get('/v1/fine_tuning/jobs')
    async def list_jobs(self, token: str = Header(None)):
        # await self.authorize_current_user(token)
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        save_root = os.path.join(lazyllm.config['train_target_root'], token)
        server_running_dict = self._read_user_job_info(token)
        m = lazyllm.TrainableModule('', save_root)
        valid_models, invalid_models = m.get_all_models()
        for model_id, model_path in valid_models:
            job_id = model_path[len(save_root):].lstrip(os.sep).split(os.sep)[0]
            if job_id in server_running_dict and server_running_dict[job_id]['status'] != 'Done':
                server_running_dict[job_id]['status'] = 'Done'
                server_running_dict[job_id]['fine_tuned_model'] = model_path
            elif job_id not in server_running_dict:
                server_running_dict[job_id] = {
                    'status': 'Done',
                    'model_id': model_id,
                    'fine_tuned_model': model_path,
                }
        for model_id, model_path in invalid_models:
            job_id = model_path[len(save_root):].lstrip(os.sep).split(os.sep)[0]
            if job_id in server_running_dict and server_running_dict[job_id]['status'] == 'Done':
                server_running_dict[job_id]['status'] = 'Failed'
                server_running_dict[job_id]['fine_tuned_model'] = model_path
            elif job_id not in server_running_dict:
                server_running_dict[job_id] = {
                    'status': 'Failed',
                    'model_id': model_id,
                    'fine_tuned_model': model_path,
                }
        return server_running_dict

    @app.get('/v1/fine_tuning/jobs/{job_id}')
    async def get_job_info(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)

        return self._read_user_job_info(token, job_id)

    @app.get('/v1/fine_tuning/jobs/{job_id}/events')
    async def get_job_log(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        if info['log_path']:
            return {'log': info['log_path']}
        else:
            return {'log': 'invalid'}
