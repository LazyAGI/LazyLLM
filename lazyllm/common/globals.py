import threading
import copy
from typing import Any, Tuple
from pydantic import BaseModel as struct
from .common import package, kwargs, deprecated

class Globals(object):
    __global_attrs__ = dict(chat_history=[], global_parameters={}, trace=[], err=None)

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


class LazyLlmRequest(struct):
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


class ReqResHelper(object):
    def __init__(self):
        self.trace = ''
        self.parameters = dict()

    def make_request(self, *args, **kw):
        if len(args) == 1:
            input = args[0]
            if isinstance(input, LazyLlmRequest):
                if len(input.global_parameters) != 0:
                    assert len(self.parameters) == 0, 'Cannot set global_parameters twice!'
                    self.parameters = input.global_parameters
                kw.update(input.kwargs)
                input = input.input
            elif isinstance(input, (tuple, list)):
                for i in input:
                    assert not isinstance(i, LazyLlmRequest), 'Cannot process list of Requests'
                input = type(input)(i for i in input)
        else:
            # bind args for flow
            _is_req = [isinstance(a, LazyLlmRequest) for a in args]
            if any(_is_req):
                assert _is_req.count(True) == 1, f'More than one Request found in args: {args}'
                idx = _is_req.index(True)
                req = args[idx]
                assert not isinstance(req.input, package) or len(req.input) == 1
                args = list(args)
                args[idx] = req.input[0] if isinstance(req.input, package) else req.input
                if not self.parameters: self.parameters = req.global_parameters
                kw.update(req.kwargs)
            input = package(args)

        if isinstance(input, kwargs):
            assert len(kw) == 0, 'kwargs are provided twice.'
            kw = dict(input)
            input = tuple()
        return LazyLlmRequest(input=input, kwargs=kw, global_parameters=self.parameters)
