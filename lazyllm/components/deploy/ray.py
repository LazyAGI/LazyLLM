import time
import random

import lazyllm
from lazyllm import launchers, LazyLLMCMD
from .base import LazyLLMDeployBase


lazyllm.config.add('num_gpus_per_node', int, 8, 'NUM_GPUS_PER_NODE')

def reallocate_launcher(launcher):
    if not isinstance(launcher, (launchers.ScoLauncher, launchers.SlurmLauncher, launchers.RemoteLauncher)):
        return [], launcher
    nnode = launcher.nnode
    ngpus = launcher.ngpus
    if nnode <= 1 and ngpus <= lazyllm.config['num_gpus_per_node']:
        return [], launcher
    if nnode > 1 and ngpus <= lazyllm.config['num_gpus_per_node']:
        return [launcher.__class__(ngpus=ngpus, sync=False) for _ in range(nnode - 1)], \
            launcher.__class__(ngpus=ngpus, sync=False)
    else:
        lazyllm.LOG.warning(
            f'The number of GPUs({ngpus}) in a single node exceeds the upper '
            f"limit{(lazyllm.config['num_gpus_per_node'])}. Please check the actual "
            'number of GPUs in a single node and set the environment variable: LAZYLLM_NUM_GPUS_PER_NODE. '
            'Now LazyLLM will reconfigure the number of nodes and GPUs')
        nnode = nnode if nnode > 0 else 1  # avoid 0
        total_gpus = nnode * ngpus
        nnode = (total_gpus + lazyllm.config['num_gpus_per_node'] - 1) // lazyllm.config['num_gpus_per_node']
        last_ngpus = total_gpus % lazyllm.config['num_gpus_per_node'] or lazyllm.config['num_gpus_per_node']
        return [launcher.__class__(ngpus=lazyllm.config['num_gpus_per_node'], sync=False) for _ in range(nnode - 1)], \
            launcher.__class__(ngpus=last_ngpus, sync=False)

class Distributed(LazyLLMDeployBase):

    def __init__(self, launcher=launchers.remote(ngpus=1), master_ip=None, port=None):
        super().__init__(launcher=launcher)
        self.port = port or random.randint(30000, 40000)
        self.master_ip = master_ip

    def cmd(self):
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
            return '{ip}:{port}'
        else:
            if not self.master_ip:
                self.master_ip = f'{job.get_jobip()}:{self.port}'
            return f'{job.get_jobip()}:{self.port}'
