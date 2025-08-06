import os
import time
import uuid
import string
import random
import asyncio
import threading
import requests
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from fastapi import Body, HTTPException, Header, Query
from async_timeout import timeout
import re
import shutil
from urllib.parse import urlparse

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.llms.utils import uniform_sft_dataset
from lazyllm import FastapiApp as app
from ..services import ServerBase

DEFAULT_TOKEN = "default_token"

def is_url(path):
    return bool(re.match(r'^https?://', path))


class Dataset(BaseModel):
    dataset_download_uri: str
    format: int
    dataset_id: str


class JobDescription(BaseModel):
    name: str
    model: str
    training_args: dict = Field(default_factory=dict)
    training_dataset: List[Dataset] = []
    validation_dataset: List[Dataset] = []
    validate_dataset_split_percent: float = Field(default=0.0)
    stage: str = ""
    num_gpus: int = 1

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

    @app.post('/v1/finetuneTasks')
    async def create_job(self, job: JobDescription = None, finetune_task_id: str = Query(None),  token: str = Header(DEFAULT_TOKEN)):
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        # Build Job-ID:
        create_time = datetime.now().strftime(self._time_format)
        job_id = '-'.join(['ft', create_time, str(uuid.uuid4())[:5]])

        # Build Model-ID:
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=7))
        model_id = job.name + '_' + random_string

        # Build checkpoint save dir:
        # - No-Env-Set: (work/path + save_ckpt) + token + job_id;
        # - Env-Set:    (train_target_root)     + token + job_id;
        save_root = os.path.join(lazyllm.config['train_target_root'], token, job_id)

        # Add launcher into hyperparameters:
        hypram = job.training_args
        hypram['ngpus'] = job.num_gpus

        # Uniform Training DataSet:
        assert len(job.training_dataset) == 1, "just support one train dataset"
        data_path = job.training_dataset[0].dataset_download_uri
        data_path = '/home/mnt/dengyuang/workspace/LazyLLM/train_data_for_code_alpace_20k.json'
        if is_url(data_path):
            parsed_url = urlparse(data_path)
            from urllib.parse import parse_qs
            query_params = parse_qs(parsed_url.query)
            if 'filename' in query_params:
                filename_param = query_params['filename'][0]
                filename = os.path.basename(filename_param)
            
            if not filename:
                filename = 'downloaded_data.json'
            local_path = os.path.join(lazyllm.config['data_path'], filename)
            
            resp = requests.get(data_path)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                shutil.copyfileobj(resp.raw, f)
            data_path = local_path
        data_path = uniform_sft_dataset(data_path, target='alpaca')

        # Set params for TrainableModule:
        m = lazyllm.TrainableModule(job.model, save_root)\
            .trainset(data_path)\
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
            'finetune_task_id': job_id,
            'base_model': job.model,
            'created_at': create_time,
            'fine_tuned_model': save_path,
            'status': status,
            'data_path': data_path,
            'hyperparameters': hypram,
            'log_path': log_path,
            'started_at': started_time,
            'cost': None,
        })

        return {'finetune_task_id': job_id, 'status': status}

    @app.delete('/v1/fine_tuning/jobs/{job_id}')
    async def cancel_job(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):
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
    async def list_jobs(self, token: str = Header(DEFAULT_TOKEN)):
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

    @app.get('/v1/finetuneTasks/{job_id}')
    async def get_job_info(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)

        return self._read_user_job_info(token, job_id)

    @app.get('/v1/finetuneTasks/{job_id}/log')
    async def get_job_log(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        if info['log_path']:
            with open(info['log_path'], 'r') as f:
                log_content = f.read()
            return log_content
        else:
            raise HTTPException(status_code=404, detail='日志路径不存在')
        
    @app.post('/v1/finetuneTasks/{job_id}:pause')
    def pause_job(self, job_id: str, name: str = Body(), token: str = Header(DEFAULT_TOKEN)):
        return
    
    @app.post('/v1/finetuneTasks/{job_id}:resume')
    def resume_job(self, job_id: str, name: str = Body(), token: str = Header(DEFAULT_TOKEN)):
        return
    
    @app.get('/v1/finetuneTasks/{job_id}runningMetrics')
    def get_running_metrics(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):
        return
        
    @app.get('/v1/models:all')
    def get_support_model(self, token: str = Header(DEFAULT_TOKEN)):
        return []
