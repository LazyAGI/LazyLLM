from .base import LazyLLMDeployBase
from .relay import RelayServer, FastapiApp
from .lightllm import Lightllm  # noqa NID002
from .vllm import Vllm  # noqa NID002
from .lmdeploy import LMDeploy
from .infinity import Infinity
from .ray import Distributed
from .mindie import Mindie
from .embed import EmbeddingDeploy
from .stable_diffusion import StableDiffusionDeploy
from .text_to_speech import TTSDeploy, BarkDeploy, ChatTTSDeploy, MusicGenDeploy
from .speech_to_text import SenseVoiceDeploy
from .ocr import OCRDeploy


__all__ = [
    'LazyLLMDeployBase',
    'RelayServer',
    'FastapiApp',
    'Lightllm',
    'Vllm',
    'LMDeploy',
    'Mindie',
    'Infinity',
    'Distributed',
    'EmbeddingDeploy',
    'StableDiffusionDeploy',
    'TTSDeploy',
    'BarkDeploy',
    'ChatTTSDeploy',
    'MusicGenDeploy',
    'SenseVoiceDeploy',
    'OCRDeploy',
]
