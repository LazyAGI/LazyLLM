import lazyllm
from lazyllm import LOG
from .trainablemodule import TrainableModule
from .onlineChatModule import OnlineChatModule

class AutoModel:
    """用于部署在线 API 模型或本地模型的模块，支持加载在线推理模块或本地可微调模块。

Args:
    model (str): 指定要加载的模型名称，例如 ``internlm2-chat-7b``，可为空。为空时默认加载 ``internlm2-chat-7b``。
    source (str): 指定要使用的在线模型服务，如需使用在线模型，必须传入此参数。支持 ``qwen`` / ``glm`` / ``openai`` / ``moonshot`` 等。
    framework (str): 指定本地部署所使用的推理框架，支持 ``lightllm`` / ``vllm`` / ``lmdeploy``。将通过 ``TrainableModule`` 与指定框架组合进行部署。
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
