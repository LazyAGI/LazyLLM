import os
import time
import uuid
import copy
import asyncio
import threading
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException, Header
from async_timeout import timeout
from starlette.responses import StreamingResponse

import lazyllm
from lazyllm.launcher import Status
from lazyllm import FastapiApp as app
from lazyllm import FileSystemQueue

class JobDescription(BaseModel):
    deploy_model: str = Field(default="qwen1.5-0.5b-chat")
    num_gpus: int = Field(default=1)
    deploy_parameters: dict = Field(
        default={
            'dtype': 'auto',
            'kv-cache-dtype': 'auto',
            'tokenizer-mode': 'auto',
            'device': 'auto',
            'block-size': 16,
            'tensor-parallel-size': 1,
            'seed': 0,
            'port': 'auto',
            'host': '0.0.0.0',
            'max-num-seqs': 256,
        }
    )

class InferServer:

    def __init__(self):
        self._user_job_deploy_info = {'default': dict()}
        self._active_job_deploy = dict()
        self._info_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._time_format = '%y%m%d%H%M%S%f'
        self._polling_thread = None
        self.pool = lazyllm.ThreadPoolExecutor(max_workers=50)

    def __call__(self):
        if not self._polling_thread:
            self._polling_status_checker()

    def __reduce__(self):
        return (self.__class__, ())

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
            if isinstance(dict_value, tuple):  # for self._active_job_deploy
                dicts[k1][k2] = dict_value
            elif isinstance(dict_value, dict):  # for self._user_job_deploy_info
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

    def _update_user_job_deploy_info(self, token, job_id=None, dict_value=None):
        self._update_dict(self._info_lock, self._user_job_deploy_info, token, job_id, dict_value)

    def _update_active_job_deploy(self, token, job_id=None, dict_value=None):
        self._update_dict(self._active_lock, self._active_job_deploy, token, job_id, dict_value)

    def _read_user_job_deploy_info(self, token, job_id=None, key=None):
        return self._read_dict(self._info_lock, self._user_job_deploy_info, token, job_id, key)

    def _read_active_job_deploy(self, token, job_id=None):
        return self._read_dict(self._active_lock, self._active_job_deploy, token, job_id, deepcopy=False)

    def _in_user_job_deploy_info(self, token, job_id=None, key=None):
        return self._in_dict(self._info_lock, self._user_job_deploy_info, token, job_id, key)

    def _in_active_job_deploy(self, token, job_id=None):
        return self._in_dict(self._active_lock, self._active_job_deploy, token, job_id)

    def _pop_user_job_deploy_info(self, token, job_id=None, key=None):
        return self._pop_dict(self._info_lock, self._user_job_deploy_info, token, job_id, key)

    def _pop_active_job_deploy(self, token, job_id=None):
        return self._pop_dict(self._active_lock, self._active_job_deploy, token, job_id)

    def _update_status(self, token, job_id):
        if not self._in_active_job_deploy(token, job_id):
            return
        # Get basic info
        info = self._read_user_job_deploy_info(token, job_id)

        # Get status
        m, _ = self._read_active_job_deploy(token, job_id)
        status = m.status().name
        log_path = info['log_path']

        update = {'status': status}

        # Some tasks not run when they are just created
        if Status[status] == Status.Running and not info['started_at']:
            update['started_at'] = datetime.now().strftime(self._time_format)

        # Ready to Infer
        if Status[status] == Status.Running and m._url:
            update['status'] = 'Ready'

        # Some tasks cannot obtain the storage path when they are just started
        if not log_path:
            save_root = os.path.join(lazyllm.config['infer_log_root'], token, job_id)
            update['log_path'] = self._get_log_path(save_root)

        # Update Status
        self._update_user_job_deploy_info(token, job_id, update)

        # Pop and kill jobs with status: Done, Failed
        if Status[status] in (Status.Done, Status.Failed):
            m, _ = self._pop_active_job_deploy(token, job_id)
            m.stop()
            if info['started_at'] and not info['cost']:
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_deploy_info(token, job_id, {'cost': cost})
            return

        create_time = datetime.strptime(info['created_at'], self._time_format)
        delta_time = (datetime.now() - create_time).total_seconds()

        # More than 5 min pop and kill jobs with status: Cancelled. Because of
        # some tasks have just been started and their status cannot be checked.
        if delta_time > 300 and Status[status] == Status.Cancelled:
            m, _ = self._pop_active_job_deploy(token, job_id)
            m.stop()
            if info['started_at'] and not info['cost']:
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_deploy_info(token, job_id, {'cost': cost})
            return

        # More than 50 min pop and kill jobs with status: TBSubmitted, InQueue, Pending
        if delta_time > 3000 and Status[status] in (Status.TBSubmitted, Status.InQueue, Status.Pending):
            m, _ = self._pop_active_job_deploy(token, job_id)
            m.stop()
            return

    def _get_log_path(self, log_dir):
        if not log_dir:
            return None

        log_files_paths = []
        for file in os.listdir(log_dir):
            if file.endswith(".log") and file.startswith("infer_"):
                log_files_paths.append(os.path.join(log_dir, file))
        if len(log_files_paths) == 0:
            return None
        if len(log_files_paths) == 1:
            return log_files_paths[-1]
        newest_file = None
        newest_time = 0
        for path in log_files_paths:
            mtime = os.path.getmtime(path)
            if mtime > newest_time:
                newest_time = mtime
                newest_file = path
        return newest_file

    def _polling_status_checker(self, frequent=5):
        def polling():
            while True:
                # Thread-safe access to two-level keys
                with self._active_lock:
                    loop_items = [(token, job_id) for token in self._active_job_deploy.keys()
                                  for job_id in self._active_job_deploy[token]]
                # Update the status of all jobs in sequence
                for token, job_id in loop_items:
                    self._update_status(token, job_id)
                time.sleep(frequent)

        self._polling_thread = threading.Thread(target=polling)
        self._polling_thread.daemon = True
        self._polling_thread.start()

    async def authorize_current_user(self, Bearer: str = None):
        if not self._in_user_job_deploy_info(Bearer):
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )
        return Bearer

    @app.post("/v1/deploy/jobs")
    async def create_job(self, job: JobDescription, token: str = Header(None)):
        if not token:
            raise HTTPException(status_code=401, detail="Invalid token")
        # await self.authorize_current_user(token)
        if not self._in_user_job_deploy_info(token):
            self._update_user_job_deploy_info(token)
        # Build Job-ID:
        create_time = datetime.now().strftime(self._time_format)
        job_id = '-'.join(['inf', create_time, str(uuid.uuid4())[:5]])

        # Build checkpoint save dir:
        # - No-Env-Set: (work/path + infer_log) + token + job_id;
        # - Env-Set:    (infer_log_root)     + token + job_id;
        save_root = os.path.join(lazyllm.config['infer_log_root'], token, job_id)

        # Add launcher into hyperparameters:
        hypram = job.deploy_parameters
        hypram['launcher'] = lazyllm.launchers.remote(sync=False, ngpus=job.num_gpus)
        hypram['log_path'] = save_root

        # Set params for TrainableModule:
        m = lazyllm.TrainableModule(job.deploy_model).deploy_method((lazyllm.deploy.vllm, hypram))

        # Launch Deploy:
        thread = threading.Thread(target=m.update_server)
        thread.start()

        # Sleep 5s for launch cmd.
        try:
            async with timeout(5):
                while m.status() == Status.Cancelled:
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            pass

        log_path = self._get_log_path(save_root)

        # Save status
        status = m.status().name
        if Status[status] == Status.Running:
            started_time = datetime.now().strftime(self._time_format)
        else:
            started_time = None
        self._update_active_job_deploy(token, job_id, (m, thread))
        self._update_user_job_deploy_info(token, job_id, {
            "job_id": job_id,
            "base_model": job.deploy_model,
            "created_at": create_time,
            "status": status,
            "hyperparameters": hypram,
            "started_at": started_time,
            "log_path": log_path,
            "cost": None,
        })

        return {"job_id": job_id, 'status': status}

    @app.post("/v1/deploy/jobs/{job_id}/cancel")
    async def cancel_job(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_active_job_deploy(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        m, _ = self._pop_active_job_deploy(token, job_id)
        info = self._read_user_job_deploy_info(token, job_id)
        m.stop()

        total_sleep = 0
        while m.status() != Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 10:
                raise HTTPException(status_code=404, detail=f"Task {job_id}, cancelled timed out.")

        status = m.status().name
        update_dict = {'status': status}
        if info['started_at'] and not info['cost']:
            update_dict['cost'] = (datetime.now() - datetime.strptime(info['started_at'],
                                                                      self._time_format)).total_seconds()
        self._update_user_job_deploy_info(token, job_id, update_dict)

        return {"status": status}

    @app.post("/v1/infer/jobs/{job_id}")
    async def infer_llm(self, qurey: str, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_active_job_deploy(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        m, _ = self._read_active_job_deploy(token, job_id)

        if not m._url:
            raise HTTPException(status_code=503, detail="Deployment is not yet complete")

        res = m(qurey)
        return res

    @app.post("/v1/stream_infer/jobs/{job_id}")
    async def stream_infer_llm(self, query: str, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_active_job_deploy(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        m, _ = self._read_active_job_deploy(token, job_id)
        if not m._url:
            raise HTTPException(status_code=503, detail="Deployment is not yet complete")

        func_future = self.pool.submit(m, query, stream_output=True)

        def generate():
            result = ''
            while True:
                if value := FileSystemQueue().dequeue():
                    result += ''.join(value)
                    yield ''.join(value)
                elif value := FileSystemQueue.get_instance('lazy_error').dequeue():
                    result += ''.join(value)
                    yield ''.join(value)
                elif value := FileSystemQueue.get_instance('lazy_trace').dequeue():
                    result += ''.join(value)
                    yield ''.join(value)
                elif func_future.done(): break
                time.sleep(0.01)
            if FileSystemQueue().size() > 0: FileSystemQueue().clear()

        return StreamingResponse(generate(), media_type="text/plain")

    @app.get("/v1/deploy/jobs")
    async def list_jobs(self, token: str = Header(None)):
        if not self._in_user_job_deploy_info(token):
            self._update_user_job_deploy_info(token)
        server_running_dict = self._read_user_job_deploy_info(token)
        save_root = os.path.join(lazyllm.config['infer_log_root'], token)
        for job_id in os.listdir(save_root):
            if job_id not in server_running_dict:
                log_path = os.path.join(save_root, job_id)
                creation_time = os.path.getctime(log_path)
                formatted_time = datetime.fromtimestamp(creation_time).strftime(self._time_format)
                server_running_dict[job_id] = {
                    "job_id": job_id,
                    "base_model": None,
                    "created_at": formatted_time,
                    "status": 'Done',
                    "hyperparameters": None,
                    "log_path": self._get_log_path(log_path),
                    "cost": None,
                }
        return server_running_dict

    @app.get("/v1/deploy/jobs/{job_id}")
    async def get_job_info(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_user_job_deploy_info(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        self._update_status(token, job_id)

        return self._read_user_job_deploy_info(token, job_id)

    @app.get("/v1/deploy/jobs/{job_id}/events")
    async def get_job_log(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_user_job_deploy_info(token, job_id):
            raise HTTPException(status_code=404, detail="Job not found")

        self._update_status(token, job_id)
        info = self._read_user_job_deploy_info(token, job_id)

        if info['log_path']:
            return {"log": info['log_path']}
        else:
            return {"log": 'invalid'}
