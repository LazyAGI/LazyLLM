import lazyllm
from lazyllm import LOG
from .module import TrainableModule
from .onlineChatModule import OnlineChatModule

class AutoModel:
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
            model = model or "internlm2-chat-7b"
            return TrainableModule(model).deploy_method(getattr(lazyllm.deploy, framework))
        elif not model:
            try:
                return OnlineChatModule()
            except KeyError as e:
                LOG.warning("`OnlineChatModule` creation failed, and will try to "
                            f"load model internlm2-chat-7b with local `TrainableModule`. Since the error: {e}")
                return TrainableModule("internlm2-chat-7b")
        else:
            return TrainableModule(model)
