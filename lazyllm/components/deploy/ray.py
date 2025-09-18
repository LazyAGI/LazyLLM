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
            f'limit{(lazyllm.config["num_gpus_per_node"])}. Please check the actual '
            'number of GPUs in a single node and set the environment variable: LAZYLLM_NUM_GPUS_PER_NODE.')
        lazyllm.LOG.error(erro_info)
        raise RuntimeError(erro_info)

class Distributed(LazyLLMDeployBase):
    """Distributed deployment class, inherits from LazyLLMDeployBase.

Provides distributed model deployment functionality based on Ray framework, supports multi-node cluster deployment.

Args:
    launcher: Launcher configuration, defaults to remote launcher(ngpus=1)
    port (int, optional): Service port number, defaults to random port(30000-40000)

Attributes:
    finetuned_model: Fine-tuned model path
    base_model: Base model path
    master_ip: Master node IP address

Methods:
    cmd(finetuned_model, base_model, master_ip): Generate deployment command
    geturl(job): Get deployed service URL address
"""

    def __init__(self, launcher=launchers.remote(ngpus=1), port=None):  # noqa B008
        super().__init__(launcher=launcher)
        self.port = port or random.randint(30000, 40000)
        self.finetuned_model = None
        self.base_model = None
        self.master_ip = None

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        """Generate Ray distributed deployment command.

Generate corresponding Ray startup command based on whether it is a master node, supports both head node and worker node modes.

Args:
    finetuned_model: Fine-tuned model path
    base_model: Base model path
    master_ip: Master node IP address, if empty starts as head node

Returns:
    LazyLLMCMD: Object containing deployment command
"""
        self.finetuned_model = finetuned_model
        self.base_model = base_model
        self.master_ip = master_ip
        if not self.master_ip:
            cmd = f'ray start --block --head --port={self.port} && sleep 365d'
        else:
            cmd = f'ray start --address={self.master_ip} && sleep 365d'
        return LazyLLMCMD(cmd=cmd, return_value=self.geturl)

    def geturl(self, job=None):
        """Get URL address of distributed deployment service.

Return corresponding service address information based on deployment mode, supports display mode and actual deployment mode.

Args:
    job: Job object, defaults to current job

Returns:
    Package: Packaged object containing model path and service address
"""
        time.sleep(5)
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return lazyllm.package(self.finetuned_model, self.base_model, None)
        else:
            if self.master_ip:
                return lazyllm.package(self.finetuned_model, self.base_model, self.master_ip)
            return lazyllm.package(self.finetuned_model, self.base_model, f'{job.get_jobip()}:{self.port}')
