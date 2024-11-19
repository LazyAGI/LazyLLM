import os
import time
import uuid
import copy
import string
import random
import threading
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException, Header

import lazyllm
from lazyllm.launcher import Status
from lazyllm.module.utils import uniform_sft_dataset
from lazyllm import FastapiApp as app


class JobCreate(BaseModel):
    finetune_model_name: str
    base_model: str = Field(default="qwen1.5-0.5b-chat")
    data_path: str = Field(default="alpaca/alpaca_data_zh_128.json")
    hyperparameters: dict = Field(
        default={
            "stage": "sft",
            "finetuning_type": "lora",
            "val_size": 1,
            "num_train_epochs": 1,
            "learning_rate": 0.0001,
            "lr_scheduler_type": "cosine",
            "per_device_train_batch_size": 16,
            "cutoff_len": 1024,
            "lora_r": 8,
            "lora_alpha": 32,
        }
    )

class TrainServer:

    def __init__(self):
        self._big_data = {'default': dict()}
        self._run_jobs = dict()
        self._big_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._time_format = '%y%m%d%H%M%S%f'
        self._polling_thread = None

    def __call__(self):
        if not self._polling_thread:
            self._polling_status_checker()

    @classmethod
    def rebuild(cls,):
        return cls()

    def __reduce__(self):
        return TrainServer.rebuild, ()

    def _update_dict(sef, lock, dicts, k1, k2=None, dict_value=None):
        with lock:
            if k1 not in dicts:
                dicts[k1] = {}
            if k2 is None:
                return
            if k2 not in dicts[k1]:
                dicts[k1][k2] = {}
            if dict_value is None:
                return
            if isinstance(dict_value, tuple):  # for self._run_jobs
                dicts[k1][k2] = dict_value
            elif isinstance(dict_value, dict):  # for self._big_data
                dicts[k1][k2].update(dict_value)
            else:
                raise RuntimeError('dict_value only supported: dict and tuple')

    def _read_dict(self, lock, dicts, k1=None, k2=None, vk=None, deepcopy=True):
        with lock:
            if k1 and k2 and vk:
                return copy.deepcopy(dicts[k1][k2][vk]) if deepcopy else dicts[k1][k2][vk]
            elif k1 and k2:
                return copy.deepcopy(dicts[k1][k2]) if deepcopy else dicts[k1][k2]
            elif k1:
                return copy.deepcopy(dicts[k1]) if deepcopy else dicts[k1]
            else:
                raise RuntimeError('At least specific k1.')

    def _in_dict(self, lock, dicts, k1, k2=None, vk=None):
        with lock:
            if k1 not in dicts:
                return False

            if k2 is not None:
                if k2 not in dicts[k1]:
                    return False
            else:
                return True

            if vk is not None:
                if vk not in dicts[k1][k2]:
                    return False
            return True

    def _pop_dict(self, lock, dicts, k1, k2=None, vk=None):
        with lock:
            if k1 and k2 and vk:
                return dicts[k1][k2].pop(vk)
            elif k1 and k2:
                return dicts[k1].pop(k2)
            elif k1:
                return dicts.pop(k1)
            else:
                raise RuntimeError('At least specific k1.')

    def _update_big_data(self, token, job_id=None, dict_value=None):
        self._update_dict(self._big_lock, self._big_data, token, job_id, dict_value)

    def _update_run_jobs(self, token, job_id=None, dict_value=None):
        self._update_dict(self._run_lock, self._run_jobs, token, job_id, dict_value)

    def _read_big_data(self, token, job_id=None, key=None):
        return self._read_dict(self._big_lock, self._big_data, token, job_id, key)

    def _read_run_jobs(self, token, job_id=None):
        return self._read_dict(self._run_lock, self._run_jobs, token, job_id, deepcopy=False)

    def _in_big_data(self, token, job_id=None, key=None):
        return self._in_dict(self._big_lock, self._big_data, token, job_id, key)

    def _in_run_jobs(self, token, job_id=None):
        return self._in_dict(self._run_lock, self._run_jobs, token, job_id)

    def _pop_big_data(self, token, job_id=None, key=None):
        return self._pop_dict(self._big_lock, self._big_data, token, job_id, key)

    def _pop_run_jobs(self, token, job_id=None):
        return self._pop_dict(self._run_lock, self._run_jobs, token, job_id)

    def _update_status(self, token, job_id):
        if not self._in_run_jobs(token, job_id):
            return
        # Get basic info
        info = self._read_big_data(token, job_id)
        save_path = info['fine_tuned_model']
        log_path = info['log_path']

        # Get status
        m, _ = self._read_run_jobs(token, job_id)
        status = m.status(info['model_id']).name

        update = {'status': status}
        # Some tasks cannot obtain the storage path when they are just started
        if not save_path:
            update['fine_tuned_model'] = self._get_save_path(m)
        if not log_path:
            update['log_path'] = self._get_log_path(m)

        # Update Status
        self._update_big_data(token, job_id, update)

        # Pop and kill jobs with status: Done, Failed
        if Status[status] in (Status.Done, Status.Failed):
            m, _ = self._pop_run_jobs(token, job_id)
            m.stop(info['model_id'])
            return

        start_time = datetime.strptime(info['created_at'], self._time_format)
        delta_time = (datetime.now() - start_time).total_seconds()

        # More than 5 min pop and kill jobs with status: Cancelled. Because of
        # some tasks have just been started and their status cannot be checked.
        if delta_time > 300 and Status[status] == Status.Cancelled:
            m, _ = self._pop_run_jobs(token, job_id)
            m.stop(info['model_id'])
            return

        # More than 50 min pop and kill jobs with status: TBSubmitted, InQueue, Pending
        if delta_time > 3000 and Status[status] in (Status.TBSubmitted, Status.InQueue, Status.Pending):
            m, _ = self._pop_run_jobs(token, job_id)
            m.stop(info['model_id'])
            return

    def _get_save_path(self, model):
        if not hasattr(model._impl, '_finetuned_model_path'):
            return None
        return model._impl._finetuned_model_path

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
            if file.endswith(".log") and file.startswith("train_log_"):
                log_files_paths.append(os.path.join(log_dir, file))
        if len(log_files_paths) == 0:
            return None
        assert len(log_files_paths) == 1
        return log_files_paths[-1]

    def _polling_status_checker(self, frequent=5):
        def polling():
            while True:
                # Thread-safe access to two-level keys
                with self._run_lock:
                    loop_items = [(token, job_id) for token in self._run_jobs.keys()
                                  for job_id in self._run_jobs[token]]
                # Update the status of all jobs in sequence
                for token, job_id in loop_items:
                    self._update_status(token, job_id)
                time.sleep(frequent)

        self._polling_thread = threading.Thread(target=polling)
        self._polling_thread.daemon = True
        self._polling_thread.start()

    async def authorize_current_user(self, Bearer: str = None):
        if not self._in_big_data(Bearer):
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )
        return Bearer

    @app.post("/register_bearer")
    async def register_bearer(self, bearer: str = None):
        if not bearer:
            return HTTPException(status_code=400, detail="Bearer token is required.")
        if not self._in_big_data(bearer):
            self._update_big_data(bearer)
            return {"message": "Bearer registered successfully."}
        else:
            return HTTPException(status_code=409, detail="Bearer already registered.")

    @app.post("/v1/fine_tuning/jobs")
    async def create_job(self, job: JobCreate, token: str = Header(None)):
        await self.authorize_current_user(token)
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
        save_root = lazyllm.config['train_target_root'] if lazyllm.config['train_target_root'] \
            else os.path.join(os.getcwd(), 'save_ckpt')
        save_root = os.path.join(save_root, token, job_id)

        # Add launcher into hyperparameters:
        hypram = job.hyperparameters
        hypram['launcher'] = lazyllm.launcher.RemoteLauncher(sync=False, ngpus=1)

        # Uniform Training DataSet:
        job.data_path = os.path.join(lazyllm.config['data_path'], job.data_path)
        job.data_path = uniform_sft_dataset(job.data_path, target='alpaca')

        # Set params for TrainableModule:
        m = lazyllm.TrainableModule(job.base_model, save_root)\
            .mode('finetune')\
            .trainset(job.data_path)\
            .finetune_method((lazyllm.finetune.llamafactory, hypram))

        # Register launcher wtih model_id:
        m._impl._launchers['manual'][model_id] = hypram.pop('launcher')

        # Launch Training:
        thread = threading.Thread(target=m._update, kwargs={'mode': ['train']})
        thread.start()

        # Sleep 5s for launch cmd.
        total_sleep = 0
        while m.status(model_id) == Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 5:
                break

        # The first getting the path may be invalid, and it will be getted with each update.
        save_path = self._get_save_path(m)
        log_path = self._get_log_path(m)

        # Save status
        status = m.status(model_id).name
        self._update_run_jobs(token, job_id, (m, thread))
        self._update_big_data(token, job_id, {
            "model_id": model_id,
            "job_id": job_id,
            "base_model": job.base_model,
            "created_at": create_time,
            "fine_tuned_model": save_path,
            "status": status,
            "data_path": job.data_path,
            "hyperparameters": hypram,
            "log_path": log_path,
        })

        return {"job_id": job_id, 'status': status}

    @app.post("/v1/fine_tuning/jobs/{job_id}/cancel")
    async def cancel_job(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_run_jobs(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        m, _ = self._pop_run_jobs(token, job_id)
        info = self._read_big_data(token, job_id)
        m.stop(info['model_id'])

        total_sleep = 0
        while m.status(info['model_id']) != Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 10:
                raise HTTPException(status_code=404, detail=f"Task {job_id}, ccancelled timed out.")

        status = m.status(info['model_id']).name
        self._update_big_data(token, job_id, {'status': status})

        return {"status": status}

    @app.get("/v1/fine_tuning/jobs")
    async def list_jobs(self, token: str = Header(None)):
        await self.authorize_current_user(token)
        return self._read_big_data(token)

    @app.get("/v1/fine_tuning/jobs/{job_id}")
    async def get_job_info(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_big_data(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        self._update_status(token, job_id)

        return self._read_big_data(token, job_id)

    @app.get("/v1/fine_tuning/jobs/{job_id}/events")
    async def get_job_log(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_big_data(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        self._update_status(token, job_id)
        info = self._read_big_data(token, job_id)

        if info['log_path']:
            return {"log": info['log_path']}
        else:
            return {"log": 'invalid'}
