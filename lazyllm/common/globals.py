import threading
import contextvars
import copy
from typing import Any, Tuple, Optional, List, Dict
import uuid
import inspect
import builtins
from .common import package, kwargs, SingletonABCMeta
from .redis_client import redis_client
from .deprecated import deprecated
import asyncio
from .utils import obj2str, str2obj
from abc import abstractmethod


class ReadWriteLock(object):
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    class ReadLock:
        def __init__(self, rw_lock):
            self.rw_lock = rw_lock

        def __enter__(self):
            with self.rw_lock._read_ready:
                self.rw_lock._readers += 1

        def __exit__(self, exc_type, exc_value, traceback):
            with self.rw_lock._read_ready:
                self.rw_lock._readers -= 1
                if self.rw_lock._readers == 0:
                    self.rw_lock._read_ready.notify_all()

    class WriteLock:
        def __init__(self, rw_lock):
            self.rw_lock = rw_lock

        def __enter__(self):
            self.rw_lock._read_ready.acquire()
            while self.rw_lock._readers > 0:
                self.rw_lock._read_ready.wait()

        def __exit__(self, exc_type, exc_value, traceback):
            self.rw_lock._read_ready.release()

    def read_lock(self):
        return self.ReadLock(self)

    def write_lock(self):
        return self.WriteLock(self)

    def __deepcopy__(self, *args, **kw):
        return ReadWriteLock()

    def __reduce__(self):
        return ReadWriteLock, ()


class ThreadSafeDict(dict):
    def __init__(self, *args, **kw):
        super(__class__, self).__init__(*args, **kw)
        self._lock = ReadWriteLock()

    def __getitem__(self, key):
        with self._lock.read_lock():
            return super(__class__, self).__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock.write_lock():
            return super(__class__, self).__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock.read_lock():
            return super(__class__, self).__delitem__(key)

    def __contains__(self, key):
        with self._lock.read_lock():
            return super(__class__, self).__contains__(key)

    def get(self, key, __default=None):
        with self._lock.read_lock():
            return super(__class__, self).get(key, __default)

    def keys(self):
        with self._lock.read_lock():
            return super(__class__, self).keys()

    def values(self):
        with self._lock.read_lock():
            return super(__class__, self).values()

    def items(self):
        with self._lock.read_lock():
            return super(__class__, self).items()

    def update(self, *args, **kwargs):
        with self._lock.write_lock():
            return super(__class__, self).update(*args, **kwargs)

    def clear(self):
        with self._lock.write_lock():
            return super(__class__, self).clear()

    def pop(self, key, __default=None):
        with self._lock.write_lock():
            return super(__class__, self).pop(key, __default)

    def __len__(self):
        with self._lock.read_lock():
            return super(__class__, self).__len__()

    def __str__(self):
        with self._lock.read_lock():
            return super(__class__, self).__str__()

    def __repr__(self):
        with self._lock.read_lock():
            return super(__class__, self).__repr__()

    def __reduce__(self):
        with self._lock.read_lock():
            return (self.__class__, (dict(self),))


class Globals(metaclass=SingletonABCMeta):
    __global_attrs__ = ThreadSafeDict(user_id=None, chat_history={}, global_parameters={}, bind_args={},
                                      tool_delimiter='<|tool_calls|>', lazyllm_files={}, usage={}, _lazyllm_agent={})

    def __new__(cls, *args, **kw):
        if cls is not Globals: return super().__new__(cls)
        return RedisGlobals() if redis_client else MemoryGlobals()

    def __init__(self):
        self.__sid = contextvars.ContextVar('local_var')
        self._init_sid()

    def _init_sid(self, sid: Optional[str] = None):
        if sid is None:
            try:
                sid = f'aid-{hex(id(asyncio.current_task()))}'
            except Exception:
                sid = f'tid-{hex(threading.get_ident())}'
        self.__sid.set(sid)
        return sid

    @property
    def _sid(self) -> str:
        try:
            sid = self.__sid.get()
        except Exception:
            sid = self._init_sid()
        return sid

    @property
    def _data(self): return self._get_data()

    def get(self, __key: str, default: Any = None):
        try:
            return self[__key]
        except KeyError:
            return default

    def __setattr__(self, __name: str, __value: Any):
        if __name in type(self).__global_attrs__:
            self[__name] = __value
        else:
            super(__class__, self).__setattr__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        if __name in type(self).__global_attrs__:
            return self[__name]
        raise AttributeError(f'Attr {__name} not found in globals')

    @abstractmethod
    def _get_data(self, rois: Optional[List[str]] = None) -> dict: ...
    @abstractmethod
    def _update(self, d: Optional[Dict]) -> None: ...
    @abstractmethod
    def __setitem__(self, __key: str, __value: Any): ...
    @abstractmethod
    def __getitem__(self, __key: str): ...
    @abstractmethod
    def clear(self): ...
    @abstractmethod
    def _clear_all(self): ...
    @abstractmethod
    def __contains__(self, item): ...
    @abstractmethod
    def pop(self, *args, **kw): ...

    @property
    def pickled_data(self):
        return obj2str(self._data)

    def unpickle_and_update_data(self, data: Optional[str]) -> dict:
        if data: self._data.update(str2obj(data))

    def __call__(self):
        return builtins.globals()

    def __reduce__(self):
        return __class__, ()

class MemoryGlobals(Globals):
    def __init__(self):
        self.__data = ThreadSafeDict()
        super(__class__, self).__init__()

    @property
    def _sid(self) -> str:
        if (sid := super(__class__, self)._sid) not in self.__data:
            self.__data[sid] = copy.deepcopy(type(self).__global_attrs__)
        return sid

    def _get_data(self, rois: Optional[List[str]] = None) -> dict:
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
        try:
            return self._data[__key]
        except KeyError:
            raise KeyError(f'Cannot find key {__key}, current session-id is {self._sid}') from None

    def clear(self):
        self.__data.pop(self._sid, None)

    def _clear_all(self):
        self.__data.clear()

    def __contains__(self, item):
        return item in self.__data[self._sid]

    def pop(self, *args, **kw):
        return self._data.pop(*args, **kw)


class RedisGlobals(MemoryGlobals):
    def __init__(self):
        super().__init__()
        self._redis_client = redis_client['globals']

    def _get_redis_key(self, key: str):
        return f'globals:{self._sid}@{key}'

    def pickled_data(self):
        key = str(uuid.uuid4().hex)
        self._redis_client.set(self._get_redis_key(key), obj2str(self._data))
        return key

    def unpickle_and_update_data(self, data: str) -> dict:
        self._data.update(str2obj(self._redis_client.get(self._get_redis_key(data))))
        self._redis_client.delete(self._get_redis_key(data))

globals = Globals()


class Locals(MemoryGlobals):
    __global_attrs__ = ThreadSafeDict(_lazyllm_agent={})

    def __call__(self):
        return inspect.currentframe().f_back.f_locals

    def __getitem__(self, __key: str):
        try:
            return super().__getitem__(__key)
        except KeyError: pass  # avoid `During handling of the above exception` for better bug-reporting experience
        return globals[__key]

locals = Locals()


@deprecated
class LazyLlmRequest(object):
    input: Any = package()
    kwargs: Any = kwargs()
    global_parameters: dict = dict()


@deprecated
class LazyLlmResponse():
    messages: Any = None
    trace: str = ''
    err: Tuple[int, str] = (0, '')

    def __init__(self, *args, **kw): pass
    def __repr__(self): return repr(self.messages)
    def __str__(self): return str(self.messages)


@deprecated('obj2str')
def encode_request(input):
    return obj2str(input)


@deprecated('obj2str')
def decode_request(input, default=None):
    if input is None: return default
    return str2obj(input)
