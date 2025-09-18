import lazyllm
from lazyllm import LOG
from .trainablemodule import TrainableModule
from .onlinemodule import OnlineChatModule

class AutoModel:
    """A module for deploying either online API-based models or local models, supporting both online inference and locally trainable modules.

Args:
    model (str): The name of the model to load, e.g., ``internlm2-chat-7b``. If None, ``internlm2-chat-7b`` will be loaded by default.
    source (str): Specifies the online model service to use. Required when using online models. Supported values include ``qwen``, ``glm``, ``openai``, ``moonshot``, etc.
    framework (str): The local inference framework to use for deployment. Supported values are ``lightllm``, ``vllm``, and ``lmdeploy``. The model will be deployed via ``TrainableModule`` using the specified framework.
"""
    def __new__(cls, model=None, source=None, framework=None):
        if model in OnlineChatModule.MODELS:
            assert source is None
            source = model
            model = None
        assert source is None or source in OnlineChatModule.MODELS
        assert framework is None or framework in ['lightllm', 'vllm', 'lmdeploy']

        if source:
            return OnlineChatModule(model=model, source=source)
        elif framework:
            model = model or 'internlm2-chat-7b'
            return TrainableModule(model).deploy_method(getattr(lazyllm.deploy, framework))
        elif not model:
            try:
                return OnlineChatModule()
            except KeyError as e:
                LOG.warning('`OnlineChatModule` creation failed, and will try to '
                            f'load model internlm2-chat-7b with local `TrainableModule`. Since the error: {e}')
                return TrainableModule('internlm2-chat-7b')
        else:
            return TrainableModule(model)
