import time
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD
from .base import LazyLLMDeployBase


lazyllm.config.add('num_gpus_per_node', int, 8, 'NUM_GPUS_PER_NODE')

def sleep_moment(finetuned_model=None, base_model=None, master_ip=None):
    sleep_time = random.uniform(1, 3)
    time.sleep(sleep_time)
    return lazyllm.package(finetuned_model, base_model, master_ip)

def reallocate_launcher(launcher):
    if not isinstance(launcher, (launchers.ScoLauncher, launchers.SlurmLauncher)):
        return [], launcher
    nnode = launcher.nnode
    ngpus = launcher.ngpus
    if nnode <= 1 and ngpus <= lazyllm.config['num_gpus_per_node']:
        return [], launcher
    if nnode > 1 and ngpus <= lazyllm.config['num_gpus_per_node']:
        return [launcher.__class__(ngpus=ngpus, sync=False) for _ in range(nnode - 1)], \
            launcher.__class__(ngpus=ngpus, sync=False)
    else:
        erro_info = (
            f'At least 1 node is required. The number of GPUs({ngpus}) in a single node exceeds the upper '
            f"limit{(lazyllm.config['num_gpus_per_node'])}. Please check the actual "
            'number of GPUs in a single node and set the environment variable: LAZYLLM_NUM_GPUS_PER_NODE.')
        lazyllm.LOG.error(erro_info)
        raise RuntimeError(erro_info)

class Distributed(LazyLLMDeployBase):

    def __init__(self, launcher=launchers.remote(ngpus=1), port=None):
        super().__init__(launcher=launcher)
        self.port = port or random.randint(30000, 40000)
        self.finetuned_model = None
        self.base_model = None
        self.master_ip = None

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        self.finetuned_model = finetuned_model
        self.base_model = base_model
        self.master_ip = master_ip
        if not self.master_ip:
            cmd = f'ray start --block --head --port={self.port} && sleep 365d'
        else:
            cmd = f'ray start --address={self.master_ip} && sleep 365d'
        return LazyLLMCMD(cmd=cmd, return_value=self.geturl)

    def geturl(self, job=None):
        time.sleep(5)
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return lazyllm.package(self.finetuned_model, self.base_model, None)
        else:
            if self.master_ip:
                return lazyllm.package(self.finetuned_model, self.base_model, self.master_ip)
            return lazyllm.package(self.finetuned_model, self.base_model, f'{job.get_jobip()}:{self.port}')
