import os
import re
import time
import uuid
import copy
import psutil
import random
import threading
import subprocess
import multiprocessing
from enum import Enum
from queue import Queue
from datetime import datetime
from collections import defaultdict

import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD, final, LOG


class Status(Enum):
    TBSubmitted = 0,
    InQueue = 1
    Running = 2,
    Pending = 3,
    Done = 100,
    Cancelled = 101,  # TODO(wangzhihong): support cancel job
    Failed = 102,


class LazyLLMLaunchersBase(object, metaclass=LazyLLMRegisterMetaClass):
    Status = Status

    def __init__(self) -> None:
        self._id = str(uuid.uuid4().hex)

    def makejob(self, cmd):
        raise NotImplementedError

    def launch(self, *args, **kw):
        raise NotImplementedError

    def cleanup(self):
        for k, v in self.all_processes[self._id]:
            v.stop()
            LOG.info(f'killed job:{k}')
        self.all_processes.pop(self._id)
        self.wait()

    @property
    def status(self):
        if len(self.all_processes[self._id]) == 1:
            return self.all_processes[self._id][0][1].status
        elif len(self.all_processes[self._id]) == 0:
            return Status.Cancelled
        raise RuntimeError('More than one tasks are found in one launcher!')

    @property
    def log_path(self):
        if len(self.all_processes[self._id]) == 1:
            return self.all_processes[self._id][0][1].log_path
        elif len(self.all_processes[self._id]) == 0:
            return None
        raise RuntimeError('More than one tasks are found in one launcher!')

    def wait(self):
        for _, v in self.all_processes[self._id]:
            v.wait()

    def clone(self):
        new = copy.deepcopy(self)
        new._id = str(uuid.uuid4().hex)
        return new


lazyllm.launchers['Status'] = Status

lazyllm.config.add('launcher', str, 'empty', 'DEFAULT_LAUNCHER',
                   description='The default remote launcher to use if no launcher is specified.')
lazyllm.config.add('cuda_visible', bool, False, 'CUDA_VISIBLE',
                   description='Whether to set the CUDA_VISIBLE_DEVICES environment variable.')


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    def __init__(self, cmd, launcher, *, sync=True):
        assert isinstance(cmd, LazyLLMCMD)
        self._origin_cmd = cmd
        self.sync = sync
        self._launcher = launcher
        self.queue, self.jobid, self.ip, self.ps = Queue(), None, None, None
        self.output_hooks = []

    def _set_return_value(self):
        cmd = getattr(self, '_fixed_cmd', None)
        if cmd and callable(cmd.return_value):
            self.return_value = cmd.return_value(self)
        elif cmd and cmd.return_value:
            self.return_value = cmd.return_value
        else:
            self.return_value = self

    def get_executable_cmd(self, *, fixed=False):
        if fixed and hasattr(self, '_fixed_cmd'):
            LOG.info('Command is fixed!')
            return self._fixed_cmd
        cmd = self._origin_cmd
        if callable(cmd.cmd):
            cmd = cmd.with_cmd(cmd.cmd())
        self._fixed_cmd = cmd.with_cmd(self._wrap_cmd(cmd.cmd))
        return self._fixed_cmd

    # interfaces
    def stop(self): raise NotImplementedError
    @property
    def status(self): raise NotImplementedError
    def wait(self): pass
    def _wrap_cmd(self, cmd): return cmd

    def _start(self, *, fixed):
        cmd = self.get_executable_cmd(fixed=fixed)
        LOG.info(f'Command: {cmd}')
        if lazyllm.config['mode'] == lazyllm.Mode.Display: return
        self.ps = subprocess.Popen(cmd.cmd, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        self._get_jobid()
        self._enqueue_subprocess_output(hooks=self.output_hooks)

        if self.sync:
            self.ps.wait()
        else:
            self._launcher.all_processes[self._launcher._id].append((self.jobid, self))
            n = 0
            while self.status in (Status.TBSubmitted, Status.InQueue, Status.Pending):
                time.sleep(2)
                n += 1
                if n > 1800:  # 3600s
                    self._launcher.all_processes[self._launcher._id].pop()
                    LOG.error('Launch failed: No computing resources are available.')
                    break

    def restart(self, *, fixed=False):
        self.stop()
        time.sleep(2)
        self._start(fixed=fixed)

    def start(self, *, restart=3, fixed=False):
        self._start(fixed=fixed)
        if not (lazyllm.config['mode'] == lazyllm.Mode.Display or self._fixed_cmd.checkf(self)):
            if restart > 0:
                for ii in range(restart):
                    LOG.warning(f'Job failed, restarting... ({ii + 1}/{restart})')
                    self.restart(fixed=fixed)
                    if self._fixed_cmd.checkf(self): break
                else:
                    raise RuntimeError(f'Job failed after retrying {restart} times')
            else:
                raise RuntimeError('Job failed without retries')
        self._set_return_value()

    def _enqueue_subprocess_output(self, hooks=None):
        self.output_thread_event = threading.Event()

        def impl(out, queue):
            for line in iter(out.readline, b''):
                try:
                    line = line.decode('utf-8')
                except Exception:
                    try:
                        line = line.decode('gb2312')
                    except Exception:
                        pass
                if isinstance(line, str):
                    queue.put(line)
                    if hooks:
                        hooks(line) if callable(hooks) else [hook(line) for hook in hooks]
                LOG.info(f'{self.jobid}: {line.rstrip()}', )
                if self.output_thread_event.is_set():
                    break
            out.close()
        self.output_thread = threading.Thread(target=impl, args=(self.ps.stdout, self.queue))
        self.output_thread.daemon = True
        self.output_thread.start()

    def _generate_name(self):
        now = datetime.now()
        return str(hex(hash(now.strftime('%S%M') + str(random.randint(3, 2000)))))[2:10]

    def __deepcopy__(self, memo=None):
        raise RuntimeError('Cannot copy Job object')

    @property
    def log_path(self):
        match = re.search(r'tee\s+([^\s]+\.log)', self._origin_cmd.cmd)
        if match:
            return match.group(1)
        return None


@final
class EmptyLauncher(LazyLLMLaunchersBase):
    all_processes = defaultdict(list)

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, launcher, sync=sync)

        def _wrap_cmd(self, cmd):
            if self._launcher.ngpus == 0:
                return cmd
            gpus = self._launcher._get_idle_gpus()
            if gpus and lazyllm.config['cuda_visible']:
                if self._launcher.ngpus is None:
                    empty_cmd = f'export CUDA_VISIBLE_DEVICES={gpus[0]} && '
                elif self._launcher.ngpus <= len(gpus):
                    empty_cmd = 'export CUDA_VISIBLE_DEVICES=' + \
                                ','.join([str(n) for n in gpus[:self._launcher.ngpus]]) + ' && '
                else:
                    error_info = (f'Not enough GPUs available. Requested {self._launcher.ngpus} GPUs, '
                                  f'but only {len(gpus)} are available.')
                    LOG.error(error_info)
                    raise error_info
            else:
                empty_cmd = ''
            return empty_cmd + cmd

        def stop(self):
            if self.ps:
                try:
                    parent = psutil.Process(self.ps.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    LOG.warning(f'Process with PID {self.ps.pid} does not exist.')
                except psutil.AccessDenied:
                    LOG.warning(f'Permission denied when trying to kill process with PID {self.ps.pid}.')
                except Exception as e:
                    LOG.warning(f'An error occurred: {e}')

        @property
        def status(self):
            return_code = self.ps.poll()
            if return_code is None: job_status = Status.Running
            elif return_code == 0: job_status = Status.Done
            else: job_status = Status.Failed
            return job_status

        def _get_jobid(self):
            self.jobid = self.ps.pid if self.ps else None

        def get_jobip(self):
            return '127.0.0.1'

        def wait(self):
            if self.ps:
                self.ps.wait()

    def __init__(self, subprocess=False, ngpus=None, sync=True, **kwargs):
        super().__init__()
        self.subprocess = subprocess
        self.sync = sync
        self.ngpus = ngpus

    def makejob(self, cmd):
        return EmptyLauncher.Job(cmd, launcher=self, sync=self.sync)

    def launch(self, f, *args, **kw):
        if isinstance(f, EmptyLauncher.Job):
            f.start()
            return f.return_value
        elif callable(f):
            if not self.subprocess:
                return f(*args, **kw)
            else:
                LOG.info('Async execution of callable object is not supported currently.')
                p = multiprocessing.Process(target=f, args=args, kwargs=kw)
                p.start()
                p.join()
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')

    def _get_idle_gpus(self):
        try:
            order_list = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
        except Exception as e:
            LOG.warning(f'Get idle gpus failed: {e}, if you have no gpu-driver, ignor it.')
            return []
        lines = order_list.strip().split('\n')

        str_num = os.getenv('CUDA_VISIBLE_DEVICES', None)
        if str_num:
            sub_gpus = [int(x) for x in str_num.strip().split(',')]

        gpu_info = []
        for line in lines:
            index, memory_free = line.split(', ')
            if not str_num or int(index) in sub_gpus:
                gpu_info.append((int(index), int(memory_free)))
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        LOG.info('Memory left:\n' + '\n'.join([f'{item[0]} GPU, left: {item[1]} MiB' for item in gpu_info]))
        return [info[0] for info in gpu_info]

class RemoteLauncher(LazyLLMLaunchersBase):
    def __new__(cls, *args, sync=False, ngpus=1, **kwargs):
        return getattr(lazyllm.launchers, lazyllm.config['launcher'])(*args, sync=sync, ngpus=ngpus, **kwargs)
