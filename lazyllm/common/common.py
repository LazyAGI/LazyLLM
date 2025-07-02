import re
import os
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

try:
    from typing import override
except ImportError:
    def override(func: Callable):
        return func


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

    def __add__(self, __other):
        return package(super().__add__(__other))


class kwargs(dict):
    pass


class arguments(object):
    class _None: pass

    def __init__(self, args=_None, kw=_None) -> None:
        self.args = package() if args is arguments._None else args
        if not isinstance(self.args, package): self.args = package((self.args,))
        self.kw = kwargs() if kw is arguments._None else copy.copy(kw)

    def append(self, x):
        args, kw = package(), kwargs()
        if isinstance(x, package):
            args = x
        elif isinstance(x, kwargs):
            kw = x
        elif isinstance(x, arguments):
            args, kw = x.args, x.kw
        else:
            args = package((x,))
        if args: self.args += args
        if kw:
            dup_keys = set(self.kw.keys()).intersection(set(kw.keys()))
            assert len(dup_keys) == 0, f'Duplicated keys: {dup_keys}'
            self.kw.update(kw)
        return self


setattr(builtins, 'package', package)


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
        cmd = re.sub(r'\b(LAZYLLM_[A-Z0-9_]*?_(?:API|SECRET)_KEY)=\S+', r'\1=xxxxxx', self.cmd)
        if self.no_displays:
            for item in self.no_displays:
                pattern = r'(-{1,2}' + re.escape(item) + r')(\s|=|)(\S+|)'
                cmd = re.sub(pattern, "", cmd)
            return cmd
        else:
            return cmd

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

    if not config['repr_show_child']: subs = []

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
        self._exc = None
        self._reset_on_pickle = reset_on_pickle
        self._lock = threading.RLock()

    def set(self, flag=True):
        with self._lock:
            self._flag = flag

    def set_exception(self, exc):
        self._exc = exc

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

def call_once(flag: once_flag, func: Callable, *args, **kw):
    with flag._lock:
        if not flag:
            try:
                return func(*args, **kw)
            except Exception as e:
                flag.set_exception(e)
            finally:
                flag.set()
        if flag._exc:
            raise flag._exc
    return None

def once_wrapper(reset_on_pickle):
    flag = reset_on_pickle if isinstance(reset_on_pickle, bool) else False

    class Wrapper:
        class Impl:
            def __init__(self, func, instance):
                self._func, self._instance = func, instance
                flag_name = f'_lazyllm_{func.__name__}_once_flag'
                if instance and not hasattr(instance, flag_name): setattr(instance, flag_name, once_flag(flag))

            def __call__(self, *args, **kw):
                assert self._instance is not None, f'{self._func} can only be used as instance method'
                return call_once(self.flag, self._func, self._instance, *args, **kw)

            __doc__ = property(lambda self: self._func.__doc__)
            def __repr__(self): return repr(self._func)

            @__doc__.setter
            def __doc__(self, value): self._func.__doc__ = value

            @property
            def flag(self) -> once_flag:
                return getattr(self._instance, f'_lazyllm_{self._func.__name__}_once_flag')

        def __init__(self, func):
            self.__func__ = func

        def __get__(self, instance, _):
            return Wrapper.Impl(self.__func__, instance)

    return Wrapper if isinstance(reset_on_pickle, bool) else Wrapper(reset_on_pickle)


class DynamicDescriptor:
    class Impl:
        def __init__(self, func, instance, owner):
            self._func, self._instance, self._owner = func, instance, owner

        def __call__(self, *args, **kw):
            return self._func(self._instance, *args, **kw) if self._instance else self._func(self._owner, *args, **kw)

        def __repr__(self): return repr(self._func)
        __doc__ = property(lambda self: self._func.__doc__)

        @__doc__.setter
        def __doc__(self, value): self._func.__doc__ = value

    def __init__(self, func):
        self.__func__ = func

    def __get__(self, instance, owner):
        return DynamicDescriptor.Impl(self.__func__, instance, owner)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def reset_on_pickle(*fields):
    def decorator(cls):
        original_getstate = cls.__getstate__ if hasattr(cls, '__getstate__') else lambda self: self.__dict__
        original_setstate = (cls.__setstate__ if hasattr(cls, '__setstate__') else
                             lambda self, state: self.__dict__.update(state))

        def __getstate__(self):
            state = original_getstate(self).copy()
            for field, *_ in fields:
                state[field] = None
            return state

        def __setstate__(self, state):
            original_setstate(self, state)
            for field in fields:
                field, field_type = field if isinstance(field, (tuple, list)) else (field, None)
                if field in state and state[field] is None and field_type is not None:
                    setattr(self, field, field_type() if field_type else None)

        cls.__getstate__ = __getstate__
        cls.__setstate__ = __setstate__
        return cls
    return decorator

class EnvVarContextManager:
    def __init__(self, env_vars_dict):
        self.env_vars_dict = {var: value for var, value in env_vars_dict.items() if value is not None}
        self.original_values = {}

    def __enter__(self):
        for var, value in self.env_vars_dict.items():
            if var in os.environ:
                self.original_values[var] = os.environ[var]
            os.environ[var] = value
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for var in self.env_vars_dict:
            if var in self.original_values:
                os.environ[var] = self.original_values[var]
            else:
                del os.environ[var]
