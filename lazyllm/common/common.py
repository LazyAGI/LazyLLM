import re
import builtins
import typing
from typing import Any, Callable
from contextlib import contextmanager
import copy
import threading
import types
from ..configs import config

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

class CaseInsensitiveDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        for key, value in dict(*args, **kwargs).items():
            assert isinstance(key, str)
            self[key] = value

    def __getitem__(self, key):
        assert isinstance(key, str)
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        super().__setitem__(key.lower(), value)

    def __contains__(self, key):
        assert isinstance(key, str)
        return super().__contains__(key.lower())

# pack return value of modules used in pipeline / parallel.
# will unpack when passing it to the next item.
class package(tuple):
    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, types.GeneratorType)):
            return super(__class__, cls).__new__(cls, args[0])
        else:
            return super(__class__, cls).__new__(cls, args)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return package(super(__class__, self).__getitem__(key))
        return super(__class__, self).__getitem__(key)


class kwargs(dict):
    pass


class arguments(object):
    class __None: pass

    def __init__(self, args=__None, kw=__None) -> None:
        self.args = package() if args is arguments.__None else args
        self.kw = kwargs() if args is arguments.__None else kw


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
            if self._item_key is not Bind.Input.__None: return input[self._item_key]
            elif self._attr_key is not Bind.Input.__None: return getattr(input, self._attr_key)
            return input

    def __init__(self, __bind_func=__None, *args, **kw):
        self._f = __bind_func() if isinstance(__bind_func, type) and __bind_func is not Bind.__None else __bind_func
        self._args = args
        self._kw = kw
        self._has_root = any([isinstance(a, AttrTree) for a in args])
        self._has_root = self._has_root or any([isinstance(v, AttrTree) for k, v in kw.items()])

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


class Identity():
    def __init__(self, *args, **kw):
        pass

    def __call__(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return package(*inputs)

    def __repr__(self):
        return make_repr('Module', 'Identity')


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
        if config['repr_ml']:
            sub_cate = re.split('>| ', subs[0][1:])[0]
            subs = rreplace(subs[0], f'</{sub_cate}>', f'</{category}>', 1)
        else:
            subs = subs[0]
        return repr[:-1] + f' sub-category={subs[1:]}'

    # ident
    sub_repr = []
    for idx, value in enumerate(subs):
        for i, v in enumerate(value.strip().split('\n')):
            if not config['repr_ml']:
                if idx != len(subs) - 1:
                    sub_repr.append(f' |- {v}' if i == 0 else f' |  {v}')
                else:
                    sub_repr.append(f' └- {v}' if i == 0 else f'    {v}')
            else:
                sub_repr.append(f'    {v}')
    if len(sub_repr) > 0: repr += ('\n' + '\n'.join(sub_repr) + '\n')
    if config['repr_ml']: repr += f'</{category}>'
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
        self._lock = threading.RLock()

    def set(self, flag=True):
        with self._lock:
            self._flag = flag

    def reset(self):
        self.set(False)

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
            flag.set()
            return func(*args, **kw)
    return None

def once_wrapper(reset_on_pickle):
    flag = reset_on_pickle if isinstance(reset_on_pickle, bool) else False

    def impl(func):
        flag_name = f'_lazyllm_{func.__name__}_once_flag'

        def wrapper(self, *args, **kw):
            if not hasattr(self, flag_name): setattr(self, flag_name, once_flag(flag))
            wrapper.flag = getattr(self, flag_name)
            return call_once(wrapper.flag, func, self, *args, **kw)

        return wrapper

    return impl if isinstance(reset_on_pickle, bool) else impl(reset_on_pickle)
