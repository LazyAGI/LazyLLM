import os
import time
import asyncio
import threading
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException, Header  # noqa NID002
from async_timeout import timeout

import lazyllm
from lazyllm.launcher import Status
from lazyllm import FastapiApp as app
from ..services import ServerBase

DEFAULT_TOKEN = 'default_token'

class _JobDescription(BaseModel):
    service_name: str
    model_name: str = Field(default='qwen1.5-0.5b-chat')
    framework: str = Field(default='auto')
    num_gpus: int = Field(default=1)


class InferServer(ServerBase):

    def _update_status(self, token, job_id):  # noqa: C901
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
            update['endpoint'] = m._url

        # Some tasks cannot obtain the storage path when they are just started
        if not log_path:
            save_root = os.path.join(lazyllm.config['infer_log_root'], token, job_id)
            os.makedirs(save_root, exist_ok=True)
            update['log_path'] = self._get_log_path(save_root)

        if Status[status] == Status.Cancelled:
            first_seen = info.get('first_cancelled_time')
            if not first_seen:
                update['first_cancelled_time'] = datetime.now().strftime(self._time_format)
                update['status'] = 'Pending'
            else:
                first_seen_time = datetime.strptime(first_seen, self._time_format)
                if (datetime.now() - first_seen_time).total_seconds() > 60:  # Observe for 60 seconds
                    ret = self._pop_active_job(token, job_id)
                    if ret is not None:
                        ret[0].stop()
                    if info['started_at'] and not info['cost']:
                        cost = (first_seen_time - datetime.strptime(info['started_at'],
                                                                    self._time_format)).total_seconds()
                        update['cost'] = cost
                    self._update_user_job_info(token, job_id, update)
                    return
                else:
                    # Still in the obsesrvation period, not cleaned up
                    update['status'] = 'Pending'
        else:
            # The status is restored, clear first_cancelled_time
            if 'first_cancelled_time' in info:
                update['first_cancelled_time'] = None

        # Update Status
        self._update_user_job_info(token, job_id, update)

        # Pop and kill jobs with status: Done, Failed
        if Status[status] in (Status.Done, Status.Failed):
            ret = self._pop_active_job(token, job_id)
            if ret is not None:
                ret[0].stop()
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

    @app.post('/v1/inference_services')
    async def create_job(self, job: _JobDescription, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        if self._in_active_jobs(token, job.service_name):
            raise HTTPException(status_code=400, detail='Service name already exists')

        job_id = job.service_name
        create_time = datetime.now().strftime(self._time_format)

        # Build checkpoint save dir:
        # - No-Env-Set: (work/path + infer_log) + token + job_id;
        # - Env-Set:    (infer_log_root)     + token + job_id;
        save_root = os.path.join(lazyllm.config['infer_log_root'], token, job_id)
        os.makedirs(save_root, exist_ok=True)
        # wait 5 minutes for launch cmd
        hypram = dict(launcher=lazyllm.launchers.remote(sync=False, ngpus=job.num_gpus, retry=30), log_path=save_root, tp=job.num_gpus)
        m = lazyllm.TrainableModule(job.model_name).deploy_method((getattr(lazyllm.deploy, job.framework), hypram))

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
        if Status[status] == Status.Cancelled:
            first_seen = datetime.now().strftime(self._time_format)
            status = 'Pending'
        else:
            first_seen = None
        self._update_active_jobs(token, job_id, (m, thread))
        self._update_user_job_info(token, job_id, {
            'lwsName': job_id,
            'status': status,
            'endpoint': 'unknown',
            'service_name': job.service_name,
            'model_name': job.model_name,
            'created_at': create_time,
            'hyperparameters': hypram,
            'started_at': started_time,
            'log_path': log_path,
            'cost': None,
            'deploy_method': m._deploy_type.__name__,
            'first_cancelled_time': first_seen,
        })

        return {'lwsName': job_id}

    @app.delete('/v1/inference_services/{job_id}')
    async def cancel_job(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
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

    @app.get('/v1/inference_services')
    async def list_jobs(self, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        if not self._in_user_job_info(token):
            self._update_user_job_info(token)
        server_running_dict = self._read_user_job_info(token)
        return server_running_dict

    @app.get('/v1/inference_services/{job_id}')
    async def get_job_info(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)

        return self._read_user_job_info(token, job_id)

    @app.get('/v1/inference_services/{job_id}/events')
    async def get_job_log(self, job_id: str, token: str = Header(DEFAULT_TOKEN)):  # noqa B008
        await self.authorize_current_user(token)
        if not self._in_user_job_info(token, job_id):
            raise HTTPException(status_code=404, detail='Job not found')

        self._update_status(token, job_id)
        info = self._read_user_job_info(token, job_id)

        if info['log_path']:
            return {'log': info['log_path']}
        else:
            return {'log': 'invalid'}
