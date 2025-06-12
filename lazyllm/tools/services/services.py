import copy
import time
import threading
from fastapi import HTTPException

class ServerBase(object):
    def __init__(self):
        self._user_job_info = {'default': dict()}
        self._active_jobs = dict()
        self._info_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._time_format = '%y%m%d%H%M%S%f'
        self._polling_thread = None

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
            if isinstance(dict_value, tuple):  # for self._active_jobs
                dicts[k1][k2] = dict_value
            elif isinstance(dict_value, dict):  # for self._user_job_info
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

    def _update_user_job_info(self, token, job_id=None, dict_value=None):
        self._update_dict(self._info_lock, self._user_job_info, token, job_id, dict_value)

    def _update_active_jobs(self, token, job_id=None, dict_value=None):
        self._update_dict(self._active_lock, self._active_jobs, token, job_id, dict_value)

    def _read_user_job_info(self, token, job_id=None, key=None):
        return self._read_dict(self._info_lock, self._user_job_info, token, job_id, key)

    def _read_active_job(self, token, job_id=None):
        return self._read_dict(self._active_lock, self._active_jobs, token, job_id, deepcopy=False)

    def _in_user_job_info(self, token, job_id=None, key=None):
        return self._in_dict(self._info_lock, self._user_job_info, token, job_id, key)

    def _in_active_jobs(self, token, job_id=None):
        return self._in_dict(self._active_lock, self._active_jobs, token, job_id)

    def _pop_user_job_info(self, token, job_id=None, key=None):
        return self._pop_dict(self._info_lock, self._user_job_info, token, job_id, key)

    def _pop_active_job(self, token, job_id=None):
        return self._pop_dict(self._active_lock, self._active_jobs, token, job_id)

    def _update_status(self, token, job_id): pass

    def _polling_status_checker(self, frequent=5):
        def polling():
            while True:
                # Thread-safe access to two-level keys
                with self._active_lock:
                    loop_items = [(token, job_id) for token in self._active_jobs.keys()
                                  for job_id in self._active_jobs[token]]
                # Update the status of all jobs in sequence
                for token, job_id in loop_items:
                    self._update_status(token, job_id)
                time.sleep(frequent)

        self._polling_thread = threading.Thread(target=polling)
        self._polling_thread.daemon = True
        self._polling_thread.start()

    async def authorize_current_user(self, Bearer: str = None):
        if not self._in_user_job_info(Bearer):
            raise HTTPException(
                status_code=401,
                detail='Invalid token',
            )
        return Bearer
