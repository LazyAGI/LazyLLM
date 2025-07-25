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
    """此类是 ``LazyLLMDeployBase`` 的子类，可根据输入的参数自动选择合适的推理框架和参数，以对大语言模型进行推理。

具体而言，基于输入的：``base_model`` 的模型参数、``max_token_num``、``launcher`` 中GPU的类型以及卡数，该类可以自动选择出合适的推理框架（如: ``Lightllm`` 或 ``Vllm``）及所需的参数。

Args:
    base_model (str): 用于进行微调的基模型，要求是基模型的路径或模型名。用于提供基模型信息。
    source (lazyllm.config['model_source']): 指定模型的下载源。可通过设置环境变量 ``LAZYLLM_MODEL_SOURCE`` 来配置，目前仅支持 ``huggingface`` 或 ``modelscope`` 。若不设置，lazyllm不会启动自动模型下载。
    trust_remote_code (bool): 是否允许加载来自远程服务器的模型代码，默认为 ``True``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    stream (bool): 是否为流式响应，默认为 ``False``。
    type (str): 类型参数，默认为 ``None``，及``llm``类型，另外还支持``embed``类型。
    max_token_num (int): 输入微调模型的token最大长度，默认为``1024``。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。注意这里能够指定的关键字参数取决于 LazyLLM 推测出的框架，因此建议谨慎设置。



Examples:
    >>> from lazyllm import deploy
    >>> deploy.auto('internlm2-chat-7b')
    <lazyllm.llm.deploy type=Lightllm> 
    """

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
        base_model = ModelManager(source).download(base_model) or ''
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
