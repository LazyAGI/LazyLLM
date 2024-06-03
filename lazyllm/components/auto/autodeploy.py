import lazyllm
from lazyllm import launchers, deploy
from ..deploy.base import LazyLLMDeployBase
from .configure import get_configer
from .auto_helper import model_map, get_model_name, check_requirements


class AutoDeploy(LazyLLMDeployBase):
    message_format = {}
    input_key_name = None
    default_headers = {'Content-Type': 'application/json'}

    def __new__(cls, base_model, trust_remote_code=True, max_token_num=1024,
                launcher=launchers.remote, stream=False, **kw):
        model_name = get_model_name(base_model)
        map_name = model_map(model_name)
        candidates = get_configer().query_deploy(lazyllm.config['gpu_type'], launcher.ngpus,
                                                 map_name, max_token_num)

        for c in candidates:
            if check_requirements(c.framework.lower()):
                deploy_cls = getattr(deploy, c.framework.lower())
            for key, value in deploy_cls.auto_map.items():
                if value:
                    kw[value] = getattr(c, key)
            return deploy_cls(trust_remote_code=trust_remote_code, launcher=launcher, stream=stream, **kw)
        raise RuntimeError(f'No valid framework found, candidates are {[c.framework.lower() for c in candidates]}')
