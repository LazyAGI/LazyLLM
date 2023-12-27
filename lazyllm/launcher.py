from typing import Any
import lazyllm
from lazyllm import LazyLLMRegisterMetaClass
from enum import Enum
import os

class LazyLLMLaunchersBase(object, metaclass=LazyLLMRegisterMetaClass):
    class Handle(object):
        pass

    def launch(self, f, *args, **kw) -> Handle:
        raise NotImplementedError


class Status(Enum):
    TBSubmitted = 0,
    InQueue = 1
    Running = 2,
    Done = 100,
    Cancelled = 101, # TODO(wangzhihong): support cancel job
    Failed = 102,


lazyllm.launchers['Status'] = Status


class EmptyLauncher(LazyLLMLaunchersBase):
    def launch(self, f, *args, **kw):
        if isinstance(f, str):
            os.system(f)
        else:
            return f(*args, **kw)


class SubprocessLauncher(LazyLLMLaunchersBase):
    # TODO(wangzhihong): support return value
    def launch(self, f, *args, **kw) -> None:
        if isinstance(f, str):
            os.system(f)
        else:
            import multiprocessing
            p = multiprocessing.Process(target=f, args=args, kwargs=kw)
            p.start()
            p.join()


class SlurmLauncher(LazyLLMLaunchersBase):
    # TODO(wangzhihong): support configs; None -> lookup config
    def __init__(self, partition, nproc, ngpus, timeout):
        self.partition = partition
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        super(__class__, self).__init__()
    
    def launch(self, cmd) -> None:
        assert isinstance(cmd, str), 'Slurm launcher only support cmd'
        os.system(f'srun -p {self.partition} -N {self.nproc} {cmd}')


class ScoLauncher(LazyLLMLaunchersBase):
    def __init__(self, nproc, ngpus, timeout):
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        super(__class__, self).__init__()

    def launch(self, cmd) -> None:
        assert isinstance(cmd, str), 'Sco launcher only support cmd'
        os.system(f'{cmd}')
