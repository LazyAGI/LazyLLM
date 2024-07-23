from lazyllm import LOG
from .module import TrainableModule
from .onlineChatModule import OnlineChatModule

class AutoModel:
    def __new__(cls, model=None):
        if not model:
            try:
                chat = OnlineChatModule()
            except KeyError as e:
                LOG.warning("`OnlineChatModule` creation failed, and will try to "
                            f"load model internlm2-chat-7b with local `TrainableModule`. Since the error: {e}")
                chat = TrainableModule("internlm2-chat-7b")
        else:
            chat = TrainableModule(model)
        return chat
