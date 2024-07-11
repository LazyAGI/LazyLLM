import threading
import contextvars
import copy
from typing import Any, Tuple, Optional, List, Dict
import pickle
from pydantic import BaseModel as struct
from .common import package, kwargs
from .deprecated import deprecated
import asyncio
import base64

class Globals(object):
    __global_attrs__ = dict(chat_history={}, global_parameters={}, trace=[], err=None)

    def __init__(self):
        self.__data = {}
        self.__sid = contextvars.ContextVar('local_var')
        self._init_sid()

    def _init_sid(self, sid: Optional[str] = None):
        if sid is None:
            try:
                sid = f'aid-{hex(id(asyncio.current_task()))}'
            except Exception:
                sid = f'tid-{hex(threading.get_ident())}'
        self.__sid.set(sid)

    @property
    def _sid(self) -> str:
        try:
            return self.__sid.get()
        except Exception:
            self._init_sid()
        return self.__sid.get()

    @property
    def _data(self): return self._get_data()

    def _get_data(self, rois: Optional[List[str]] = None) -> dict:
        if self._sid not in self.__data:
            self.__data[self._sid] = copy.deepcopy(__class__.__global_attrs__)
        if rois:
            assert isinstance(rois, (tuple, list))
            return {k: v for k, v in self.__data[self._sid].items() if k in rois}
        return self.__data[self._sid]

    def _update(self, d: Optional[Dict]) -> None:
        if d:
            self._data.update(d)

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
        if __name in __class__.__global_attrs__:
            return self[__name]
        raise AttributeError(f'Attr {__name} not found in globals')

    def clear(self):
        self.__data.pop(self._sid)

    def _clear_all(self):
        self.__data.clear()

globals = Globals()


@deprecated
class LazyLlmRequest(object):
    input: Any = package()
    kwargs: Any = kwargs()
    global_parameters: dict = dict()


@deprecated
class LazyLlmResponse(struct):
    messages: Any = None
    trace: str = ''
    err: Tuple[int, str] = (0, '')

    def __repr__(self): return repr(self.messages)
    def __str__(self): return str(self.messages)


def encode_request(input):
    return base64.b64encode(pickle.dumps(input)).decode('utf-8')


def decode_request(input, default=None):
    if input is None: return default
    return pickle.loads(base64.b64decode(input.encode('utf-8')))
