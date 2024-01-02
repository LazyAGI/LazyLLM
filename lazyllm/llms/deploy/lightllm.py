import lazyllm
from lazyllm import launchers, flows, package
from .base import LazyLLMDeployBase
from ..core import register

@register('deploy')
def lightllm_stage1(base_model, lora_weights):
    print(f'deploy: base_model = {base_model}, lora_weights = {lora_weights}')
    return package([base_model, lora_weights])


@register('deploy', cmd=True)
def lightllm_stage2(base_model, lora_weights):
    return f'python deploy.py --base_model={base_model}, --lora_weights={lora_weights}'


class Lightllm(LazyLLMDeployBase, flows.NamedPipeline):
    def __init__(self, *, launcher=launchers.slurm):
        super().__init__(launcher=launcher)
        flows.NamedPipeline.__init__(self,
            deploy_stage1 = deploy.lightllm_stage1(launcher=launchers.empty),
            deploy_stage2 = flows.namedParallel(
	    	deploy_stage21 = deploy.lightllm_stage2(),
	    	deploy_stage22 = deploy.lightllm_stage2(),
	    )
	)

    def __call__(self, base_model, lora_weights=None):
        if not isinstance(base_model, package):
            base_model = package([base_model[0], base_model[1]])
        return flows.NamedPipeline.__call__(self, base_model)