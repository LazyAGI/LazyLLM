import atexit
from multiprocessing.util import register_after_fork

from .base import LazyLLMLaunchersBase, Status, Job, EmptyLauncher, RemoteLauncher
from .slurm import SlurmLauncher
from .sco import ScoLauncher
from .k8s import K8sLauncher
from lazyllm import LOG

def cleanup():
    # empty
    for m in (EmptyLauncher, SlurmLauncher, ScoLauncher, K8sLauncher):
        while m.all_processes:
            _, vs = m.all_processes.popitem()
            for k, v in vs:
                v.stop()
                LOG.info(f'killed job:{k}')
    LOG.close()

atexit.register(cleanup)

def _exitf(*args, **kw):
    atexit._clear()
    atexit.register(cleanup)

register_after_fork(_exitf, _exitf)


__all__ = ['LazyLLMLaunchersBase', 'Status', 'Job', 'EmptyLauncher',
           'SlurmLauncher', 'ScoLauncher', 'K8sLauncher', 'RemoteLauncher']
