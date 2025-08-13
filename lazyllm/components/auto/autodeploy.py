# flake8: noqa: C901
import lazyllm
import math
import shutil
import functools
import re
from lazyllm import launchers, deploy, LazyLLMLaunchersBase, config
from ..deploy.base import LazyLLMDeployBase
from .dependencies.requirements import requirements
from .auto_helper import get_model_name, check_requirements
from ..deploy.stable_diffusion.stable_diffusion3 import StableDiffusionDeploy
from ..deploy.speech_to_text.sense_voice import SenseVoiceDeploy
from ..deploy.text_to_speech import TTSDeploy
from ..deploy.ocr.pp_ocr import OCRDeploy
from ..utils.downloader import ModelManager
from typing import Optional


@functools.lru_cache
def check_cmd(framework):
    framework_mapping = dict(mindie='mindieservice_daemon')
    return bool(shutil.which(framework_mapping.get(framework, framework)))


class AutoDeploy(LazyLLMDeployBase):
    @staticmethod
    def _get_embed_deployer(launcher, type, kw):
        launcher = launcher or launchers.remote(ngpus=1)
        kw['model_type'] = type
        if lazyllm.config['default_embedding_engine'].lower() in ('transformers', 'flagembedding') \
            or kw.get('embed_type')=='sparse' or not check_requirements('infinity_emb'):
            return deploy.Rerank if type == 'reranker' else deploy.Embedding, launcher, kw
        else:
            return deploy.InfinityRerank if type == 'reranker' else deploy.Infinity, launcher, kw

    @classmethod
    def get_deployer(cls, base_model: str, source: Optional[str] = None, trust_remote_code: bool = True,
                     launcher: Optional[LazyLLMLaunchersBase] = None, type: Optional[str] = None,
                     log_path: Optional[str] = None, **kw):
        model_name = get_model_name(base_model)
        kw['log_path'], kw['trust_remote_code'] = log_path, trust_remote_code
        if not type:
            type = ModelManager.get_model_type(model_name)

        if type in ('embed', 'cross_modal_embed', 'reranker'):
            return AutoDeploy._get_embed_deployer(launcher, type, kw)
        elif type == 'sd':
            return StableDiffusionDeploy, launcher or launchers.remote(ngpus=1), kw
        elif type == 'stt':
            return SenseVoiceDeploy, launcher or launchers.remote(ngpus=1), kw
        elif type == 'tts':
            return TTSDeploy.get_deploy_cls(model_name), launcher or launchers.remote(ngpus=1), kw
        elif type == 'vlm':
            return deploy.LMDeploy, launcher or launchers.remote(ngpus=1), kw
        elif type == 'ocr':
            return OCRDeploy, launcher or launchers.remote(ngpus=1), kw

        if not launcher:
            match = re.search(r'(\d+)[bB]', model_name)
            size = int(match.group(1)) if match else 0
            size = (size * 2) if 'awq' not in model_name.lower() else (size / 1.5)
            ngpus = (1 << (math.ceil(size * 2 / config['gpu_memory']) - 1).bit_length())
            launcher = launchers.remote(ngpus = ngpus)

        for deploy_cls in ['vllm', 'lightllm', 'lmdeploy', 'mindie']:
            if check_cmd(deploy_cls) or check_requirements(requirements.get(deploy_cls)):
                deploy_cls = getattr(deploy, deploy_cls)
                return deploy_cls, launcher, kw
        return deploy.auto, launcher, kw

    def __new__(cls, base_model, source=lazyllm.config['model_source'], trust_remote_code=True,
                launcher=None, type=None, log_path=None, **kw):
        deploy_cls, launcher, kw = __class__.get_deployer(
            base_model=base_model, source=source, trust_remote_code=trust_remote_code,
            launcher=launcher, type=type, log_path=log_path, **kw)
        return deploy_cls(launcher=launcher, **kw)


config.add('gpu_memory', int, 80, 'GPU_MEMORY')
