import re
import builtins
import typing
from typing import Any, Iterable, Callable
from contextlib import contextmanager
import signal
import copy

import lazyllm

try:
    from typing import final
except ImportError:
    _F = typing.TypeVar("_F", bound=Callable[..., Any])
    def final(f: _F) -> _F: return f


# Special Dict for lazy programmer. Suppose we have a LazyDict as follows：
#    >>> ld = LazyDict(name='ld', ALd=int)
# 1. Use dot instead of ['str']
#    >>> ld.ALd
# 2. Support lowercase first character to make the sentence more like a function
#    >>> ld.aLd
# 3. Supports direct calls to dict when there is only one element
#    >>> ld()
# 4. Support dynamic default key
#    >>> ld.set_default('ALd')
#    >>> ld.default
# 5. allowed to omit the group name if the group name appears in the name
#    >>> ld.a
class LazyDict(dict):
    def __init__(self, name, *args, **kw):
        super(__class__, self).__init__(*args, **kw)
        self._default = None
        self.name = name.capitalize()

    def __setitem__(self, key, value):
        assert key != 'default', 'LazyDict do not support key: default'
        return super().__setitem__(key, value)

    # default -> self.default
    # key -> Key, keyName, KeyName
    # if self.name ends with 's' or 'es', ignor it 
    def __getattr__(self, key):
        key = self._default if key == 'default' else key
        keys = [key, f'{key[0].upper()}{key[1:]}', f'{key}{self.name}', f'{key[0].upper()}{key[1:]}{self.name}']
        if self.name.endswith('s'):
            n = 2 if self.name.endswith('es') else 1
            keys.extend([f'{key}{self.name[:-n]}', f'{key[0].upper()}{key[1:]}{self.name[:-n]}'])

        for k in keys:
            if k in self.keys():
                return self[k]
        # return super(__class__, self).__getattribute__(key)
        raise AttributeError(f'Attr {key} not found in {self}')

    def __call__(self, *args, **kwargs):
        assert self._default is not None or len(self.keys()) == 1
        return self.default if self._default else self[list(self.keys())[0]](*args, **kwargs)

    def set_default(self, key):
        assert isinstance(key, str), 'default key must be str'
        self._default = key
        

class LazyLLMRegisterMetaClass(type):
    all_clses = dict()
    all_groups = dict()

    def __new__(metas, name, bases, attrs):
        new_cls = type.__new__(metas, name, bases, attrs)
        if name.startswith('LazyLLM') and name.endswith('Base'):
            group = re.match('(LazyLLM)(.*)(Base)', name.split('.')[-1])[2].lower()
            assert not hasattr(new_cls, '_lazy_llm_group')
            new_cls._lazy_llm_group = group

            LazyLLMRegisterMetaClass.all_clses.update({group:LazyDict(group)})
            LazyLLMRegisterMetaClass.all_groups.update({group:new_cls})

            assert not hasattr(builtins, group), f'group name \'{group}\' cannot be used'
            setattr(builtins, group, LazyLLMRegisterMetaClass.all_clses[group])
            setattr(lazyllm, group, LazyLLMRegisterMetaClass.all_clses[group])
        elif hasattr(new_cls, '_lazy_llm_group'):
            group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
            assert new_cls.__name__ not in group, (
                f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}')
            group[new_cls.__name__] = new_cls
        return new_cls


# pack return value of modules used in pipeline / parallel.
# will unpack when passing it to the next item.
class package(tuple):
    def __new__(cls, *args):
        return super(__class__, cls).__new__(cls, args[0]
            if len(args) == 1 and isinstance(args[0], Iterable) else args)


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

    def __repr__(self):
        return f'placeholder._{self.idx}'

for i in range(10):
    exec(f'_{i} = Placeholder({i})')


class Bind(object):
    def __init__(self, f, *args):
        self._f = f() if isinstance(f, type) else f
        self._args = args

    def __call__(self, *args):
        return self._f(*[args[a.idx] if isinstance(a, Placeholder) else a
                         for a in self._args])
    def __repr__(self) -> str:
        return self._f.__repr__() + '(bind args:{})'.format(
            ', '.join([repr(a) if a is not self else 'self' for a in self._args]))

    def __getattr__(self, name):
        return getattr(self._f, name)

setattr(builtins, 'bind', Bind)

class LazyLLMCMD(object):
    def __init__(self, cmd, *, return_value=None, post_function=None, no_displays=None) -> None:
        if isinstance(cmd, (tuple, list)):
            cmd = ' && '.join(cmd)
        assert isinstance(cmd, str), 'cmd must be (list of) bash command str.'
        assert return_value is None or post_function is None, \
            'Cannot support return_value and post_function at the same time'
        self.cmd = cmd
        self.return_value = return_value
        self.post_function = post_function
        self.no_displays = no_displays

    def __hash__(self):
        return hash(self.cmd)

    def __str__(self):
        if self.no_displays:
            cmd = self.cmd
            for item in self.no_displays:
                pattern = r'(-{1,2}' + re.escape(item) + r')(\s|=|)(\S+|)'
                cmd = re.sub(pattern, "", cmd)
            return cmd
        else:
            return self.cmd

@contextmanager
def timeout(duration, *,  msg=''):
    def timeout_handler(signum, frame):
        m = f'{msg}, ' if msg else msg 
        m += f'block timedout after timeout: {duration} s'
        raise TimeoutError(m)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


class ReadOnlyWrapper(object):
    def __init__(self, obj=None):
        self.obj = obj

    def set(self, obj):
        self.obj = obj

    def __getattr__(self, key):
        if self.obj is not None:
            return getattr(self.obj, key)
        return super(__class__, self).__getattr__(key)

    def __repr__(self):
        return f'{self.obj.__repr__()[:-1]}(Readonly)>'
