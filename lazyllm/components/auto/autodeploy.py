import lazyllm
from lazyllm import launchers, deploy, LOG
from ..deploy.base import LazyLLMDeployBase
from .configure import get_configer
from .auto_helper import model_map, get_model_name, check_requirements
from lazyllm.components.embedding.embed import EmbeddingDeploy
from ..utils.downloader import ModelDownloader

class AutoDeploy(LazyLLMDeployBase):
    message_format = {}
    input_key_name = None
    default_headers = {'Content-Type': 'application/json'}

    def __new__(cls, base_model, source=lazyllm.config['model_source'], trust_remote_code=True, max_token_num=1024,
                launcher=launchers.remote(ngpus=1), stream=False, type=None, **kw):
        base_model = ModelDownloader(source).download(base_model)
        model_name = get_model_name(base_model)
        if type == 'embed' or cls.get_model_type(model_name) == 'embed':
            return EmbeddingDeploy(trust_remote_code, launcher)
        map_name = model_map(model_name)
        candidates = get_configer().query_deploy(lazyllm.config['gpu_type'], launcher.ngpus,
                                                 map_name, max_token_num)

        for c in candidates:
            if check_requirements(c.framework.lower()):
                deploy_cls = getattr(deploy, c.framework.lower())
            if c.tgs <= 0: LOG.warning(f"Model {model_name} may out of memory under Framework {c.framework}")
            for key, value in deploy_cls.auto_map.items():
                if value:
                    kw[value] = getattr(c, key)
            return deploy_cls(trust_remote_code=trust_remote_code, launcher=launcher, stream=stream, **kw)
        raise RuntimeError(f'No valid framework found, candidates are {[c.framework.lower() for c in candidates]}')

    @classmethod
    def get_model_type(cls, model_name):
        from lazyllm.components.utils.downloader.model_mapping import model_name_mapping
        if model_name in model_name_mapping:
            return model_name_mapping[model_name].get('type', 'llm')
        else:
            return 'llm'
