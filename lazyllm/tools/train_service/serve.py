import os
import time
import uuid
import string
import random
import asyncio
import json
import threading
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from fastapi import Body, HTTPException, Header, Query
from async_timeout import timeout
import re
import shutil
from fastapi.responses import StreamingResponse
import requests
from urllib.parse import unquote

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.llms.utils import uniform_sft_dataset
from lazyllm import FastapiApp as app
from ..services import ServerBase

DEFAULT_TOKEN = 'default_token'

def is_url(path):
    return bool(re.match(r'^https?://', path))

def get_filename_from_url(url: str, timeout: int = 10) -> str:
    """
    Get filename from URL Content-Disposition header
    """
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        cd = resp.headers.get("Content-Disposition", "")
        for pattern in [r"filename\*=UTF-8''(.+)", r'filename="?([^"]+)"?']:
            match = re.search(pattern, cd)
            if match:
                return unquote(match.group(1))
    except Exception:
        pass

    raise HTTPException(status_code=404, detail='File format is not clear')


class Dataset(BaseModel):
    dataset_download_uri: str
    format: int
    dataset_id: str

class TrainingArgs(BaseModel):
    val_size: float = 0.02
    num_train_epochs: int = 1
    learning_rate: float = 0.1
    lr_scheduler_type: str = 'cosine'
    per_device_train_batch_size: int = 32
    cutoff_len: int = 1024
    finetuning_type: str = 'lora'
    lora_rank: int = 8
    lora_alpha: int = 32
    trust_remote_code: bool = True
    ngpus: int = 1

    class Config:
        extra = "allow"  # extra fields are allowed

class JobDescription(BaseModel):
    name: str
    model: str
    training_args: TrainingArgs = Field(default_factory=TrainingArgs)
    training_dataset: List[Dataset] = []
    validation_dataset: List[Dataset] = []
    validate_dataset_split_percent: float = Field(default=0.0)
    stage: str = ""

class ModelExport(BaseModel):
    name: str
    model_display_name: str
    model_id: str


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
            update['log_path'] = m.log_path(info['model_id'])

        # Update Status
        self._update_user_job_info(token, job_id, update)

        # Pop and kill jobs with status: Failed
        if Status[status] == Status.Failed:
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

    @app.post('/v1/finetuneTasks')
    async def create_job(self, job: JobDescription, finetune_task_id: str = Query(None),  # noqa B008
                         token: str = Header(DEFAULT_TOKEN)):  # noqa B008
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
        os.makedirs(save_root, exist_ok=True)

        # Add launcher into hyperparameters:
        hypram = job.training_args.model_dump()

        # Uniform Training DataSet:
        assert len(job.training_dataset) == 1, "just support one train dataset"
        data_path = job.training_dataset[0].dataset_download_uri
        if is_url(data_path):
            response = requests.get(data_path, stream=True)
            if response.status_code == 200:
                file_name = get_filename_from_url(data_path)
                target_path = os.path.join(save_root, file_name)
                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                data_path = target_path
            else:
                raise HTTPException(status_code=404, detail='dataset download failed')

        data_path = uniform_sft_dataset(data_path, target='alpaca')

        # Set params for TrainableModule:
        m = lazyllm.TrainableModule(job.model, save_root)\
            .trainset(data_path)\
            .finetune_method(lazyllm.finetune.llamafactory)

        # Launch Training:
        thread = threading.Thread(target=m._impl._async_finetune, args=(model_id,), kwargs=hypram)
        thread.start()
        await asyncio.sleep(1)

        # Sleep 5s for launch cmd.
        try:
            async with timeout(5):
                while m.status(model_id) == Status.Cancelled:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            pass

        # The first getting the path may be invalid, and it will be getted with each update.
        save_path = self._get_save_path(m)
        log_path = m.log_path(model_id)

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

    @app.delete('/v1/finetuneTasks/{job_id}')
    async def cancel_job(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
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

    @app.get('/v1/finetuneTasks/jobs')
    async def list_jobs(self, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
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
    async def get_job_info(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)

        return self._read_user_job_info(token, job_id)

    @app.get('/v1/finetuneTasks/{job_id}/log')
    async def get_job_log(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        if not info['log_path'] or not os.path.exists(info['log_path']):
            raise HTTPException(status_code=404, detail='log file not found')

        async def generate_log_stream():
            with open(info['log_path'], 'r') as f:
                for line in f:
                    if line.strip():
                        res = json.dumps({'result': {'log_data': line.strip()}})
                        yield f"data: {res}\n\n"
            yield "data: [DONE]"

        return StreamingResponse(generate_log_stream(), media_type="text/event-stream")

    @app.post('/v1/finetuneTasks/{job_id}/model:export')
    async def export_model(self, job_id: str, model: ModelExport, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        model_path = info['fine_tuned_model']
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail='model file not found')
        target_dir = os.path.join(lazyllm.config['model_path'], model.model_display_name)
        if os.path.exists(target_dir):
            raise HTTPException(status_code=404, detail='target dir already exists')
        shutil.copytree(model_path, target_dir)
        shutil.rmtree(model_path)
        return

    @app.get('/v1/finetuneTasks/{job_id}/runningMetrics')
    async def get_running_metrics(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        raise HTTPException(status_code=404, detail='not implemented')

    @app.post('/v1/finetuneTasks/{job_id}:pause')
    async def pause_job(self, job_id: str, name: str = Body(embed=True), token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        raise HTTPException(status_code=404, detail='not implemented')

    @app.post('/v1/finetuneTasks/{job_id}:resume')
    async def resume_job(self, job_id: str, name: str = Body(embed=True), token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        raise HTTPException(status_code=404, detail='not implemented')
