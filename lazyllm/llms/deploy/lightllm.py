import lazyllm
from lazyllm import launchers, flows, package, LazyLLMCMD
from .base import LazyLLMDeployBase
from ..core import register
try:
    from builtins import deploy
except Exception:
    from lazyllm import LazyLLMRegisterMetaClass
    deploy = LazyLLMRegisterMetaClass.all_groups['deploy']

@register('deploy')
def lightllm_stage1(base_model, lora_weights):
    print(f'deploy: base_model = {base_model}, lora_weights = {lora_weights}')
    return package(base_model, lora_weights)


@register('deploy', cmd=True)
def lightllm_stage2(base_model, lora_weights):
    return LazyLLMCMD(
        f'python deploy.py --base_model={base_model}, --lora_weights={lora_weights}',
        return_value='deploy_stage2')


def lightllm_stage3(s21, s22):
    print(f'deploy stage3: s21={s21}, s22={s22}')
    return None


class Lightllm(LazyLLMDeployBase, flows.NamedPipeline):
    def __init__(self, base_url, port=None, *, launcher=launchers.slurm):
        super().__init__(launcher=launcher)
        self.base_url = base_url
        self.port = port
        flows.NamedPipeline.__init__(self,
            deploy_stage1 = deploy.lightllm_stage1(launcher=launchers.empty),
            deploy_stage2 = flows.namedParallel(
	    	    deploy_stage21 = deploy.lightllm_stage2,
	    	    deploy_stage22 = deploy.lightllm_stage2,
	        ),
            deploy_stage3 = lightllm_stage3,
	)

    def __call__(self, base_model, lora_weights=None):
        if not isinstance(base_model, package):
            base_model = package(base_model[0], base_model[1])
        flows.NamedPipeline.__call__(self, base_model)
        return package(self.base_url, self.port)