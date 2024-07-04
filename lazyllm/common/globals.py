import threading
import copy
from typing import Any

class Globals(object):
    __global_attrs__ = dict(chat_history=[], global_parameters={})

    def __init__(self):
        self.__data = {}
        self.__sid = threading.local()

    @property
    def _sid(self) -> str:
        return getattr(self.__sid, 'id', f'tid-{hex(threading.get_ident())}')

    @_sid.setter
    def _sid(self, sid: str) -> None:
        self.__sid = sid

    @property
    def _data(self) -> dict:
        if self._sid not in self.__data:
            self.__data[self._sid] = copy.deepcopy(__class__.__global_attrs__)
        return self.__data[self._sid]

    def __setitem__(self, __key: str, __value: Any):
        self._data[__key] = __value

    def __getitem__(self, __key: str):
        return self._data[__key]

    def __setattr__(self, __name: str, __value: Any):
        if __name in __class__.__global_attrs__:
            self[__name] = __value
        else:
            super(__class__, self).__setattr__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        return self[__name]

globals = Globals()
