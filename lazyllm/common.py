import re
import builtins
import typing
from typing import Any, Iterable, Callable
from contextlib import contextmanager
import signal
import copy
import threading
import types
from queue import Queue

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
    def __init__(self, name='', base=None, *args, **kw):
        super(__class__, self).__init__(*args, **kw)
        self._default = None
        self.name = name.capitalize()
        self.base = base

    def __setitem__(self, key, value):
        assert key != 'default', 'LazyDict do not support key: default'
        if '.' in key:
            grp, key = key.rsplit('.', 1)
            return self[grp].__setitem__(key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        if '.' in key:
            grp, key = key.split('.', 1)
            return self[grp][key]
        return super().__getitem__(key)

    # default -> self.default
    # key -> Key, keyName, KeyName
    # if self.name ends with 's' or 'es', ignor it 
    def __getattr__(self, key):
        key = self._default if key == 'default' else key
        keys = [key, f'{key[0].upper()}{key[1:]}', f'{key}{self.name}', f'{key[0].upper()}{key[1:]}{self.name}',
                f'{key}{self.name.lower()}', f'{key[0].upper()}{key[1:]}{self.name.lower()}']
        if self.name.endswith('s'):
            n = 2 if self.name.endswith('es') else 1
            keys.extend([f'{key}{self.name[:-n]}', f'{key[0].upper()}{key[1:]}{self.name[:-n]}'])

        for k in set(keys):
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


class FlatList(list):
    def absorb(self, item):
        if isinstance(item, list):
            self.extend(item)
        elif item is not None:
            self.append(item)


group_template = '''\
class LazyLLM{name}Base(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''


class LazyLLMRegisterMetaClass(type):
    all_clses = LazyDict()

    def __new__(metas, name, bases, attrs):
        new_cls = type.__new__(metas, name, bases, attrs)
        if name.startswith('LazyLLM') and name.endswith('Base'):
            ori = re.match('(LazyLLM)(.*)(Base)', name.split('.')[-1])[2]
            group = ori.lower()
            new_cls._lazy_llm_group = '.'.join([g for g in (getattr(new_cls, '_lazy_llm_group', ''), group) if g])
            LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group] = LazyDict(group, new_cls)
            if new_cls._lazy_llm_group == group:
                for m in (builtins, lazyllm):
                    assert not (hasattr(m, group) and hasattr(m, ori)), f'group name \'{ori}\' cannot be used'
                    setattr(m, group, LazyLLMRegisterMetaClass.all_clses[group])
                    setattr(m, ori, LazyLLMRegisterMetaClass.all_clses[group])
        elif hasattr(new_cls, '_lazy_llm_group'):
            group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
            assert new_cls.__name__ not in group, (
                f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}')
            group[new_cls.__name__] = new_cls
        return new_cls


def _get_base_cls_from_registry(cls_str, *, registry=LazyLLMRegisterMetaClass.all_clses):
    if cls_str == '':
        return registry.base
    group, cls_str = cls_str.split('.', 1) if '.' in cls_str else (cls_str, '')
    if not (registry is LazyLLMRegisterMetaClass.all_clses or group in registry):
        exec(group_template.format(name=group.capitalize(), base=registry.base._lazy_llm_group))
    return _get_base_cls_from_registry(cls_str, registry=registry[group])


# pack return value of modules used in pipeline / parallel.
# will unpack when passing it to the next item.
class package(tuple):
    def __new__(cls, *args):
        return super(__class__, cls).__new__(cls, args[0]
            # Cannot use `Iterable` here because of str
            if len(args) == 1 and isinstance(args[0], (tuple, list, types.GeneratorType)) else args)


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
        # name will be '_f' in copy.deepcopy
        if name != '_f':
            return getattr(self._f, name)
        return super(__class__, self).__getattr__(name)

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
        # key will be 'obj' in copy.deepcopy
        if key != 'obj' and self.obj is not None:
            return getattr(self.obj, key)
        return super(__class__, self).__getattr__(key)

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


def ID(*inputs):
    if len(inputs) == 1:
        return inputs[0]
    return package(*inputs)