# -*- coding: utf-8 -*-

from .bark import BarkDeploy
from .chattts import ChatTTSDeploy
from .musicgen import MusicGenDeploy

class TTSDeploy:

    def __new__(cls, name, **kwarg):
        return cls.get_deploy_cls(name)(**kwarg)

    @classmethod
    def get_deploy_cls(cls, name):
        name = name.lower()
        if name == 'bark':
            return BarkDeploy
        elif name in ('chattts', 'chattts-new'):
            raise RuntimeError('ChatTTS is deprecated and no longer supported.')
            return ChatTTSDeploy
        elif name.startswith('musicgen'):
            return MusicGenDeploy
        else:
            raise RuntimeError(f'Not support model: {name}')

__all__ = [
    'TTSDeploy',
    'BarkDeploy',
    'ChatTTSDeploy',
    'MusicGenDeploy',
]
