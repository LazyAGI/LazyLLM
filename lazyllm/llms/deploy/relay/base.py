import lazyllm
from lazyllm import launchers
from ..base import LazyLLMDeployBase
import cloudpickle


class RelayServer(LazyLLMDeployBase):
    def __init__(self, pre_func= None, post_func=None, *, launcher=launchers.slurm):
        self.pre = cloudpickle.dumps(pre_func)
        self.post = cloudpickle.dumps(post_func)
        super().__init__(launcher=launcher)

    def cmd(self, url):
        cmd = f'python test2.py --target_url={url} '\
              f'--before_function="{self.pre} --after_function="{self.post}"'
        return cmd
    