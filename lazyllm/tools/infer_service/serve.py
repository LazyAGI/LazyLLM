import os
import time
import uuid
import asyncio
import threading
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException, Header
from async_timeout import timeout

import lazyllm
from lazyllm.launcher import Status
from lazyllm import FastapiApp as app
from ..services import ServerBase

class JobDescription(BaseModel):
    deploy_model: str = Field(default='qwen1.5-0.5b-chat')
    num_gpus: int = Field(default=1)


class InferServer(ServerBase):

    def _update_status(self, token, job_id):
        if not self._in_active_jobs(token, job_id):
            return
        # Get basic info
        info = self._read_user_job_info(token, job_id)

        # Get status
        m, _ = self._read_active_job(token, job_id)
        status = m.status().name
        log_path = info['log_path']

        update = {'status': status}

        # Some tasks not run when they are just created
        if Status[status] == Status.Running and not info['started_at']:
            update['started_at'] = datetime.now().strftime(self._time_format)

        # Ready to Infer
        if Status[status] == Status.Running and m._url:
            update['status'] = 'Ready'
            update['url'] = m._url

        # Some tasks cannot obtain the storage path when they are just started
        if not log_path:
            save_root = os.path.join(lazyllm.config['infer_log_root'], token, job_id)
            update['log_path'] = self._get_log_path(save_root)

        # Update Status
        self._update_user_job_info(token, job_id, update)

        # Pop and kill jobs with status: Done, Failed
        if Status[status] in (Status.Done, Status.Cancelled, Status.Failed):
            m, _ = self._pop_active_job(token, job_id)
            m.stop()
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
            m.stop()
            if info['started_at'] and not info['cost']:
                cost = (datetime.now() - datetime.strptime(info['started_at'], self._time_format)).total_seconds()
                self._update_user_job_info(token, job_id, {'cost': cost})
            return

        # More than 50 min pop and kill jobs with status: TBSubmitted, InQueue, Pending
        if delta_time > 3000 and Status[status] in (Status.TBSubmitted, Status.InQueue, Status.Pending):
            m, _ = self._pop_active_job(token, job_id)
            m.stop()
            return

    def _get_log_path(self, log_dir):
        if not log_dir:
            return None

        log_files_paths = []
        for file in os.listdir(log_dir):
            if file.endswith('.log') and file.startswith('infer_'):
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

    @app.post('/v1/deploy/jobs')
    async def create_job(self, job: JobDescription, token: str = Header(None)):
        if not token:
            raise HTTPException(status_code=401, detail='Invalid token')
        # await self.authorize_current_user(token)
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        # Build Job-ID:
        create_time = datetime.now().strftime(self._time_format)
        job_id = '-'.join(['inf', create_time, str(uuid.uuid4())[:5]])

        # Build checkpoint save dir:
        # - No-Env-Set: (work/path + infer_log) + token + job_id;
        # - Env-Set:    (infer_log_root)     + token + job_id;
        save_root = os.path.join(lazyllm.config['infer_log_root'], token, job_id)
        hypram = dict(launcher=lazyllm.launchers.remote(sync=False, ngpus=job.num_gpus), log_path=save_root)
        m = lazyllm.TrainableModule(job.deploy_model).deploy_method((lazyllm.deploy.auto, hypram))

        # Launch Deploy:
        thread = threading.Thread(target=m.start)
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
        self._update_active_jobs(token, job_id, (m, thread))
        self._update_user_job_info(token, job_id, {
            'job_id': job_id,
            'base_model': job.deploy_model,
            'created_at': create_time,
            'status': status,
            'hyperparameters': hypram,
            'started_at': started_time,
            'log_path': log_path,
            'cost': None,
            'deploy_method': m._deploy_type.__name__,
            'url': m._url,
        })

        return {'job_id': job_id, 'status': status}

    @app.post('/v1/deploy/jobs/{job_id}/cancel')
    async def cancel_job(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_active_jobs(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        m, _ = self._pop_active_job(token, job_id)
        info = self._read_user_job_info(token, job_id)
        m.stop()

        total_sleep = 0
        while m.status() != Status.Cancelled:
            time.sleep(1)
            total_sleep += 1
            if total_sleep > 10:
                raise HTTPException(status_code=404, detail=f'Task {job_id}, cancelled timed out.')

        status = m.status().name
        update_dict = {'status': status}
        if info['started_at'] and not info['cost']:
            update_dict['cost'] = (datetime.now() - datetime.strptime(info['started_at'],
                                                                      self._time_format)).total_seconds()
        self._update_user_job_info(token, job_id, update_dict)

        return {'status': status}

    @app.get('/v1/deploy/jobs')
    async def list_jobs(self, token: str = Header(None)):
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        server_running_dict = self._read_user_job_info(token)
        save_root = os.path.join(lazyllm.config['infer_log_root'], token)
        for job_id in os.listdir(save_root):
            if job_id not in server_running_dict:
                log_path = os.path.join(save_root, job_id)
                creation_time = os.path.getctime(log_path)
                formatted_time = datetime.fromtimestamp(creation_time).strftime(self._time_format)
                server_running_dict[job_id] = {
                    'job_id': job_id,
                    'base_model': None,
                    'created_at': formatted_time,
                    'status': 'Cancelled',
                    'hyperparameters': None,
                    'log_path': self._get_log_path(log_path),
                    'cost': None,
                    'deploy_method': None,
                    'url': None,
                }
        return server_running_dict

    @app.get('/v1/deploy/jobs/{job_id}')
    async def get_job_info(self, job_id: str, token: str = Header(None)):
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)

        return self._read_user_job_info(token, job_id)

    @app.get('/v1/deploy/jobs/{job_id}/events')
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
