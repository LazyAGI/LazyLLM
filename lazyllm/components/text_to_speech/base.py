
from .bark import BarkDeploy
from .chattts import ChatTTSDeploy

class TTSDeploy:

    def __new__(cls, name, **kwarg):
        if name == 'bark':
            return BarkDeploy(**kwarg)
        elif name == 'ChatTTS':
            return ChatTTSDeploy(**kwarg)
        else:
            raise RuntimeError(f"Not support model: {name}")
