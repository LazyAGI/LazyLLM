import re
import builtins
import typing
from typing import Any, Callable
from contextlib import contextmanager
import copy
import threading
import types
from queue import Queue
from pydantic import BaseModel as struct
from typing import Tuple

import lazyllm

try:
    from typing import final
except ImportError:
    _F = typing.TypeVar("_F", bound=Callable[..., Any])
    def final(f: _F) -> _F: return f


class FlatList(list):
    def absorb(self, item):
        if isinstance(item, list):
            self.extend(item)
        elif item is not None:
            self.append(item)


class ArgsDict(dict):
    def __init__(self, *args, **kwargs):
        super(ArgsDict, self).__init__(*args, **kwargs)

    def check_and_update(self, kw):
        assert set(kw.keys()).issubset(set(self)), f'unexpected keys: {set(kw.keys()) - set(self)}'
        self.update(kw)

    def parse_kwargs(self):
        string = ' '.join(f'--{k}={v}' if type(v) is not str else f'--{k}=\"{v}\"' for k, v in self.items())
        return string


# pack return value of modules used in pipeline / parallel.
# will unpack when passing it to the next item.
class package(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, types.GeneratorType)):
            return super(__class__, cls).__new__(cls, args[0])
        else:
            return super(__class__, cls).__new__(cls, args)


class kwargs(dict):
    pass


setattr(builtins, 'package', package)


class AttrTree(object):
    def __init__(self, name=None, pres=[]):
        self._path = copy.deepcopy(pres)
        if name is not None:
            self._path.append(name)

    def __str__(self):
        return '.'.join(self._path)

    def __getattr__(self, name):
        v = __class__(name, pres=self._path)
        setattr(self, name, v)
        return v

    def get_from(self, obj):
        v = obj
        for name in self._path:
            v = getattr(v, name)
        return v

    def __deepcopy__(self, memo):
        return self

root = AttrTree()


class Placeholder(object):
    _pool = dict()

    def __new__(cls, idx):
        if idx not in Placeholder._pool:
            Placeholder._pool[idx] = super().__new__(cls)
        return Placeholder._pool[idx]

    def __init__(self, idx):
        assert isinstance(idx, int)
        self.idx = idx

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return f'placeholder._{self.idx}'

for i in range(10):
    vars()[f'_{i}'] = Placeholder(i)

def _setattr(self, key, v):
    raise RuntimeError('Cannot set attr for Placeholder')
setattr(Placeholder, '__setattr__', _setattr)


class _MetaBind(type):
    def __instancecheck__(self, __instance):
        if isinstance(__instance, Bind) and isinstance(__instance._f, self):
            return True
        return super(__class__, self).__instancecheck__(__instance)


class Bind(object):
    class __None: pass

    class Input(object):
        class __None: pass

        def __init__(self): self._item_key, self._attr_key = Bind.Input.__None, Bind.Input.__None

        def __getitem__(self, key):
            self._item_key = key
            return self

        def __getattr__(self, key):
            self._attr_key = key
            return self

        def get_input(self, input):
            if isinstance(input, LazyLlmRequest):
                input = input.input if input.input else input.kwargs
            elif isinstance(input, LazyLlmResponse):
                input = input.messages
            if self._item_key is not Bind.Input.__None: return input[self._item_key]
            elif self._attr_key is not Bind.Input.__None: return getattr(input, self._attr_key)
            return input

    def __init__(self, __bind_func=__None, *args, **kw):
        self._f = __bind_func() if isinstance(__bind_func, type) and __bind_func is not Bind.__None else __bind_func
        self._args = args
        self._kw = kw
        self._has_root = any([isinstance(a, AttrTree) for a in args])

    def __ror__(self, __value: Callable):
        if self._f is not Bind.__None: self._args = (self._f,) + self._args
        self._f = __value
        return self

    # _bind_args_source: dict(input=input, args=dict(key=value))
    def __call__(self, *args, _bind_args_source=None, **kw):
        if self._f is None: return None
        keys = set(kw.keys()).intersection(set(self._kw.keys()))
        assert len(keys) == 0, f'Keys `{keys}` are already bind!'
        bind_args = args if len(self._args) == 0 else (
            [args[a.idx] if isinstance(a, Placeholder) else a for a in self._args])
        bind_kwargs = self._kw

        def get_bind_args(a):
            return a.get_input(_bind_args_source['input']) if isinstance(a, Bind.Input) else (
                _bind_args_source['args'][id(a)] if id(a) in _bind_args_source['args'] else a)

        if _bind_args_source:
            bind_args = [get_bind_args(a) for a in bind_args]
            bind_kwargs = {k: get_bind_args(v) for k, v in bind_kwargs.items()}
        return self._f(*bind_args, **bind_kwargs, **kw)

    # TODO: modify it
    def __repr__(self) -> str:
        return self._f.__repr__() + '(bind args:{})'.format(
            ', '.join([repr(a) if a is not self else 'self' for a in self._args]))

    def __getattr__(self, name):
        # name will be '_f' in copy.deepcopy
        if name != '_f':
            return getattr(self._f, name)
        return super(__class__, self).__getattr__(name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name not in ('_f', '_args', '_kw', '_has_root'):
            return setattr(self._f, __name, __value)
        return super(__class__, self).__setattr__(__name, __value)


setattr(builtins, 'bind', Bind)


class LazyLLMCMD(object):
    def __init__(self, cmd, *, return_value=None, checkf=(lambda *a: True), no_displays=None):
        if isinstance(cmd, (tuple, list)):
            cmd = ' && '.join(cmd)
        assert isinstance(cmd, str) or callable(cmd), 'cmd must be func or (list of) bash command str.'
        self.cmd = cmd
        self.return_value = return_value
        self.checkf = checkf
        self.no_displays = no_displays

    def __hash__(self):
        return hash(self.cmd)

    def __str__(self):
        assert not callable(self.cmd), f'Cannot convert cmd function {self.cmd} to str'
        if self.no_displays:
            cmd = self.cmd
            for item in self.no_displays:
                pattern = r'(-{1,2}' + re.escape(item) + r')(\s|=|)(\S+|)'
                cmd = re.sub(pattern, "", cmd)
            return cmd
        else:
            return self.cmd

    def with_cmd(self, cmd):
        # Attention: Cannot use copy.deepcopy because of class method.
        new_instance = LazyLLMCMD(cmd, return_value=self.return_value,
                                  checkf=self.checkf, no_displays=self.no_displays)
        return new_instance

    def get_args(self, key):
        assert not callable(self.cmd), f'Cannot get args from function {self.cmd}'
        pattern = r'*(-{1,2}' + re.escape(key) + r')(\s|=|)(\S+|)*'
        return re.match(pattern, self.cmd)[3]

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(duration, *, msg=''):
    def raise_timeout_exception():
        event.set()

    event = threading.Event()
    timer = threading.Timer(duration, raise_timeout_exception)
    timer.start()

    try:
        yield
    finally:
        if not event.is_set():
            timer.cancel()
        else:
            raise TimeoutException(f'{msg}, block timed out after {duration} s')


class ReadOnlyWrapper(object):
    def __init__(self, obj=None):
        self.obj = obj

    def set(self, obj):
        self.obj = obj

    def __getattr__(self, key):
        # key will be 'obj' in copy.deepcopy
        if key != 'obj' and self.obj is not None:
            return getattr(self.obj, key)
        return super(__class__, self).__getattr__(key)

    # TODO: modify it
    def __repr__(self):
        r = self.obj.__repr__()
        return (f'{r[:-1]}' if r.endswith('>') else f'<{r}') + '(Readonly)>'

    def __deepcopy__(self, memo):
        # drop obj
        return ReadOnlyWrapper()

    def isNone(self):
        return self.obj is None


class Thread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, prehook=None, daemon=None):
        self.q = Queue()
        super().__init__(group, self.work, name, (prehook, target, args), kwargs, daemon=daemon)

    def work(self, prehook, target, args):
        if prehook:
            prehook()
        try:
            r = target(*args)
        except Exception as e:
            self.q.put(e)
        else:
            self.q.put(r)

    def get_result(self):
        r = self.q.get()
        if isinstance(r, Exception):
            raise r
        return r


class Identity():
    def __init__(self, *args, **kw):
        pass

    def __call__(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return package(*inputs)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Identity')


class ResultCollector(object):
    class Impl(object):
        def __init__(self, name, value): self._name, self._value = name, value

        def __call__(self, *args, **kw):
            assert (len(args) == 0) ^ (len(kw) == 0), f'args({len(args)}), kwargs({len(kw)})'
            assert self._name is not None
            if len(args) > 0:
                self._value[self._name] = args[0] if len(args) == 1 else package(*args)
                return self._value[self._name]
            else:
                self._value[self._name] = kw
                return kwargs(kw)

    def __init__(self): self._value = dict()
    def __call__(self, name): return ResultCollector.Impl(name, self._value)
    def __getitem__(self, name): return self._value[name]
    def __repr__(self): return repr(self._value)
    def keys(self): return self._value.keys()
    def items(self): return self._value.items()


class LazyLlmRequest(struct):
    input: Any = package()
    kwargs: Any = kwargs()
    global_parameters: dict = dict()

    def split(self, flag=None):
        if flag is None:
            assert len(self.kwargs) == 0 and isinstance(self.input, (tuple, list)), (
                f'Only tuple input can be split automatically, your input is {self.input} <{type(self.input)}>')
            return [LazyLlmRequest(input=inp, global_parameters=self.global_parameters) for inp in self.input]
        elif isinstance(flag, int):
            assert len(self.kwargs) == 0 and isinstance(self.input, (tuple, list)), (
                f'Only tuple input can be split automatically, your input is {self.input} <{type(self.input)}>')
            assert flag == len(self.input), 'input size mismatch with split number'
            return [LazyLlmRequest(input=inp, global_parameters=self.global_parameters) for inp in self.input]
        elif isinstance(flag, list):
            if isinstance(self.input, dict):
                assert len(self.kwargs) == 0, 'Cannot provived input and kwargs at the same time for split'
                d = self.input
            elif isinstance(self.input, (tuple, list)):
                return self.split(len(flag))
            else:
                assert not self.input, 'Cannot provived input and kwargs at the same time for split'
                d = self.kwargs
            return [LazyLlmRequest(input=d[key], global_parameters=self.global_parameters) for key in flag]
        else: raise TypeError(f'invalid flag type {type(flag)} given')


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
            elif isinstance(input, LazyLlmResponse):
                assert len(kw) == 0
                if input.trace: self.trace += input.trace
                input = input.messages
            elif isinstance(input, (tuple, list)):
                for i in input:
                    if isinstance(i, LazyLlmResponse): self.trace += i.trace
                    else: assert not isinstance(i, LazyLlmRequest), 'Cannot process list of Requests'
                input = type(input)(i.messages if isinstance(i, LazyLlmResponse) else i for i in input)
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

    def make_response(self, res, *, force=False):
        if isinstance(res, LazyLlmResponse):
            res.trace = self.trace + res.trace
            return res
        else:
            res = res.input if isinstance(res, LazyLlmRequest) else res
            return LazyLlmResponse(messages=res, trace=self.trace) if self.trace or force else res


class ReprRule(object):
    rules = {}

    @classmethod
    def add_rule(cls, cate, type, subcate, subtype=None):
        if subtype:
            cls.rules[f'{cate}:{type}'] = f'<{subcate} type={subtype}'
        else:
            cls.rules[f'{cate}:{type}'] = f'<{subcate}'

    @classmethod
    def check_combine(cls, cate, type, subs):
        return f'{cate}:{type}' in cls.rules and subs.startswith(cls.rules[f'{cate}:{type}'])


def rreplace(s, old, new, count):
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]

def make_repr(category, type, *, name=None, subs=[], attrs=dict(), **kw):
    if len(kw) > 0:
        assert len(attrs) == 0, 'Cannot provide attrs and kwargs at the same time'
        attrs = kw

    if isinstance(type, builtins.type): type = type.__name__
    name = f' name={name}' if name else ''
    attrs = ' ' + ' '.join([f'{k}={v}' for k, v in attrs.items()]) if attrs else ''
    repr = f'<{category} type={type}{name}{attrs}>'

    if len(subs) == 1 and ReprRule.check_combine(category, type, subs[0]):
        if lazyllm.config['repr_ml']:
            sub_cate = re.split('>| ', subs[0][1:])[0]
            subs = rreplace(subs[0], f'</{sub_cate}>', f'</{category}>', 1)
        else:
            subs = subs[0]
        return repr[:-1] + f' sub-category={subs[1:]}'

    # ident
    sub_repr = []
    for idx, value in enumerate(subs):
        for i, v in enumerate(value.strip().split('\n')):
            if not lazyllm.config['repr_ml']:
                if idx != len(subs) - 1:
                    sub_repr.append(f' |- {v}' if i == 0 else f' |  {v}')
                else:
                    sub_repr.append(f' └- {v}' if i == 0 else f'    {v}')
            else:
                sub_repr.append(f'    {v}')
    if len(sub_repr) > 0: repr += ('\n' + '\n'.join(sub_repr) + '\n')
    if lazyllm.config['repr_ml']: repr += f'</{category}>'
    return repr


# if key is already in repr, then modify its value.
# if ket is not in repr, add key to repr with value.
# if value is None, remove key from repr.
def modify_repr(repr, key, value):
    # TODO: impl this function
    return repr


class once_flag(object):
    def __init__(self, reset_on_pickle=False):
        self._flag = False
        self._reset_on_pickle = reset_on_pickle
        self._lock = threading.Lock()

    def _set(self, flag=True):
        self._flag = flag

    def reset(self):
        with self._lock:
            self._set(False)

    def __bool__(self):
        return self._flag

    @classmethod
    def rebuild(cls, flag, reset_on_pickle):
        r = cls(reset_on_pickle)
        if not reset_on_pickle: r._flag = flag
        return r

    def __reduce__(self):
        return once_flag.rebuild, (self._flag, self._reset_on_pickle)

def call_once(flag, func, *args, **kw):
    with flag._lock:
        if not flag:
            flag._set()
            return func(*args, **kw)
    return None
