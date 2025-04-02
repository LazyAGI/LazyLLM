# flake8: noqa: C901
import lazyllm
from lazyllm import launchers, deploy, LOG
from ..deploy.base import LazyLLMDeployBase
from .configure import get_configer
from .dependencies.requirements import requirements
from .auto_helper import model_map, get_model_name, check_requirements
from lazyllm.components.embedding.embed import EmbeddingDeploy
from lazyllm.components.stable_diffusion.stable_diffusion3 import StableDiffusionDeploy
from lazyllm.components.speech_to_text.sense_voice import SenseVoiceDeploy
from lazyllm.components.text_to_speech.base import TTSDeploy
from ..utils.downloader import ModelManager

class AutoDeploy(LazyLLMDeployBase):
    message_format = {}
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}

    def __new__(cls, base_model, source=lazyllm.config['model_source'], trust_remote_code=True, max_token_num=1024,
                launcher=launchers.remote(ngpus=1), stream=False, type=None, log_path=None, **kw):
        base_model = ModelManager(source).download(base_model) or ''
        model_name = get_model_name(base_model)
        if not type:
            type = ModelManager.get_model_type(model_name)
        if type in ('embed', 'cross_modal_embed', 'reranker'):
            if lazyllm.config['default_embedding_engine'] == 'transformers' or lazyllm.config['default_embedding_engine'] == 'flagEmbedding' \
                or kw.get('embed_type')=='sparse' or not check_requirements('infinity_emb'):
                return EmbeddingDeploy(launcher, model_type=type, log_path=log_path, embed_type=kw.get('embed_type', 'dense'))
            else:
                return deploy.Infinity(launcher, model_type=type, log_path=log_path)
        elif type == 'sd':
            return StableDiffusionDeploy(launcher, log_path=log_path)
        elif type == 'stt':
            return SenseVoiceDeploy(launcher, log_path=log_path)
        elif type == 'tts':
            return TTSDeploy(model_name, log_path=log_path, launcher=launcher)
        elif type == 'vlm':
            return deploy.LMDeploy(launcher, stream=stream, log_path=log_path, **kw)
        map_name = model_map(model_name)
        candidates = get_configer().query_deploy(lazyllm.config['gpu_type'], launcher.ngpus,
                                                 map_name, max_token_num)

        for c in candidates:
            if check_requirements(requirements[c.framework.lower()]):
                deploy_cls = getattr(deploy, c.framework.lower())
            else:
                continue
            if c.tgs <= 0: LOG.warning(f"Model {model_name} may out of memory under Framework {c.framework}")
            for key, value in deploy_cls.auto_map.items():
                if value:
                    kw[value] = getattr(c, key)
            return deploy_cls(trust_remote_code=trust_remote_code, launcher=launcher,
                              stream=stream, log_path=log_path, **kw)
        raise RuntimeError(f'No valid framework found, candidates are {[c.framework.lower() for c in candidates]}')
