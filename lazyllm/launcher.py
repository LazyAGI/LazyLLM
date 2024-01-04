from typing import Any
import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD
from enum import Enum
import os
import time
import subprocess

class Status(Enum):
    TBSubmitted = 0,
    InQueue = 1
    Running = 2,
    Done = 100,
    Cancelled = 101, # TODO(wangzhihong): support cancel job
    Failed = 102,


class LazyLLMLaunchersBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self) -> None:
        self.status = Status.TBSubmitted

    def makejob(self, cmd):
        raise NotImplementedError

    def launch(self, *args, **kw):
        raise NotImplementedError


lazyllm.launchers['Status'] = Status


def exec_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding='utf-8', executable='/bin/bash')
    out, err = p.communicate()
    return_code = p.returncode
    assert return_code == 0, f'Exec cmd \'{cmd}\' failed, errmsg: {err}'
    return out.strip()


class EmptyLauncher(LazyLLMLaunchersBase):
    def makejob(self, cmd):
        return cmd.cmd

    def launch(self, f, *args, **kw):
        if isinstance(f, (str, tuple, list)):
            return exec_cmd(f)
        elif callable(f):
            return f(*args, **kw)
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')


class SubprocessLauncher(LazyLLMLaunchersBase):
    def makejob(self, cmd):
        return cmd.cmd

    # TODO(wangzhihong): support return value
    def launch(self, f, *args, **kw) -> None:
        if isinstance(f, (str, tuple, list)):
            return exec_cmd(f)
        elif callable(f):
            import multiprocessing
            p = multiprocessing.Process(target=f, args=args, kwargs=kw)
            p.start()
            p.join()
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    def __init__(self, cmd, *, sync=True):
        self.cmd = cmd.cmd
        self.return_value = cmd.return_value
        self.post_function = cmd.post_function
        self.sync = sync

    def get_return_value(self):
        return self.return_value if self.return_value else (
            self.post_function(self) if self.post_function else self)



class SlurmLauncher(LazyLLMLaunchersBase):
    # In order to obtain the jobid to monitor and terminate the job more
    # conveniently, only one srun command is allowed in one Job
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, sync=sync)
            self.jobname = str(hex(hash(cmd)))[2:]
            self.cmd = f'srun -p {launcher.partition} -N {launcher.nproc} bash -c \'{self.cmd}\''
            self.jobid = None
        
        def start(self):
            # exec cmd
            # get jobid
            print(self.cmd)
            pass

        def stop(self):
            # scancel job
            pass

        @property
        def status(self):
            # lookup job
            pass


    # TODO(wangzhihong): support configs; None -> lookup config
    def __init__(self, partition=None, nproc=1, ngpus=8, timeout=0, *, sync=True):
        self.partition = partition
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        self.sync = sync
        super(__class__, self).__init__()
    
    def makejob(self, cmd):
        return SlurmLauncher.Job(cmd, launcher=self)

    def get_idle_nodes(self):
        return None

    def launch(self, job) -> None:
        assert isinstance(job, SlurmLauncher.Job), 'Slurm launcher only support cmd'
        job.start()
        if self.sync:
            while job.status == Status.Running:
                time.sleep(10)
            job.stop()
        return job.get_return_value()
            

class ScoLauncher(LazyLLMLaunchersBase):
    def __init__(self, nproc, ngpus, timeout):
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        super(__class__, self).__init__()

    def launch(self, cmd) -> None:
        assert isinstance(cmd, str), 'Sco launcher only support cmd'
        os.system(f'{cmd}')
