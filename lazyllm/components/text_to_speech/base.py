
from .bark import BarkDeploy
from .chattts import ChatTTSDeploy
from .musicgen import MusicGenDeploy

class TTSDeploy:

    def __new__(cls, name, **kwarg):
        name = name.lower()
        if name == 'bark':
            return BarkDeploy(**kwarg)
        elif name in ('chattts', 'chattts-new'):
            return ChatTTSDeploy(**kwarg)
        elif name.startswith('musicgen'):
            return MusicGenDeploy(**kwarg)
        else:
            raise RuntimeError(f"Not support model: {name}")
