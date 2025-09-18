import re
import os
import builtins
import typing
from typing import Any, Callable, Optional, List, Dict
from contextlib import contextmanager
import copy
import threading
import types
from ..configs import config
from urllib.parse import urlparse

try:
    from typing import final
except ImportError:
    _F = typing.TypeVar('_F', bound=Callable[..., Any])
    def final(f: _F) -> _F:
        """A decorator to indicate final methods and final classes.

    Use this decorator to indicate to type checkers that the decorated
    method cannot be overridden, and decorated class cannot be subclassed.
    For example:

      class Base:
          @final
          def done(self) -> None:
              ...
      class Sub(Base):
          def done(self) -> None:  # Error reported by type checker
                ...

      @final
      class Leaf:
          ...
      class Other(Leaf):  # Error reported by type checker
          ...

    There is no runtime checking of these properties.
    """
        return f

try:
    from typing import override
except ImportError:
    def override(func: Callable):
        return func


class FlatList(list):
    def absorb(self, item):
        """Absorb elements into the list.

Args:
    item: Element to add, can be a single element or a list
"""
        if isinstance(item, list):
            self.extend(item)
        elif item is not None:
            self.append(item)


class ArgsDict(dict):
    """Parameter dictionary class for managing and validating command line arguments.

Args:
    *args: Positional arguments passed to parent dict class
    **kwargs: Keyword arguments passed to parent dict class

**Returns:**

- ArgsDict instance providing parameter checking and formatting functionality
"""
    def __init__(self, *args, **kwargs):
        super(ArgsDict, self).__init__(*args, **kwargs)

    def check_and_update(self, kw):
        """Check and update parameter dictionary.

Args:
    kw (dict): Parameter dictionary to update
"""
        assert set(kw.keys()).issubset(set(self)), f'unexpected keys: {set(kw.keys()) - set(self)}'
        self.update(kw)

    def parse_kwargs(self):
        """Parse parameter dictionary into command line argument string.
"""
        string = ' '.join(f'--{k}={v}' if type(v) is not str else f'--{k}="{v}"' for k, v in self.items())
        return string

class CaseInsensitiveDict(dict):
    """Case-insensitive dictionary class.

CaseInsensitiveDict inherits from dict and provides case-insensitive key-value storage and retrieval. All keys are converted to lowercase when stored, ensuring that values can be accessed regardless of whether the key name is uppercase, lowercase, or mixed case.

Features:
    - All keys are automatically converted to lowercase when stored
    - Supports standard dictionary operations (get, set, check containment)
    - Maintains all original dict functionality, only differs in key name handling

Args:
    *args: Positional arguments passed to the parent dict class
    **kwargs: Keyword arguments passed to the parent dict class


Examples:
    >>> from lazyllm.common import CaseInsensitiveDict
    >>> # 创建大小写不敏感的字典
    >>> d = CaseInsensitiveDict({'Name': 'John', 'AGE': 25, 'City': 'New York'})
    >>> 
    >>> # 使用不同大小写访问相同的键
    >>> print(d['name'])      # 使用小写
    ... 'John'
    >>> print(d['NAME'])      # 使用大写
    ... 'John'
    >>> print(d['Name'])      # 使用首字母大写
    ... 'John'
    >>> 
    >>> # 设置值时也会转换为小写
    >>> d['EMAIL'] = 'john@example.com'
    >>> print(d['email'])     # 使用小写访问
    ... 'john@example.com'
    >>> 
    >>> # 检查键是否存在（大小写不敏感）
    >>> 'AGE' in d
    True
    >>> 'age' in d
    True
    >>> 'Age' in d
    True
    >>> 
    >>> # 支持标准字典操作
    >>> d['PHONE'] = '123-456-7890'
    >>> print(d.get('phone'))
    ... '123-456-7890'
    >>> print(len(d))
    ... 5
    """
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
    """The package class is used to encapsulate the return values of pipeline or parallel modules,
ensuring automatic unpacking when passing to the next module, thereby supporting flexible multi-value passing.


Examples:
    >>> from lazyllm.common import package
    >>> p = package(1, 2, 3)
    >>> p
    (1, 2, 3)
    >>> p[1]
    2
    >>> p_slice = p[1:]
    >>> isinstance(p_slice, package)
    True
    >>> p2 = package([4, 5])
    >>> p + p2
    (1, 2, 3, 4, 5)
    """
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


builtins.package = package


class LazyLLMCMD(object):
    """Command line operation wrapper class providing secure and flexible command management.

Args:
    cmd (Union[str, List[str], Callable]):Command input, supports three formats:String command,Command list,Callable object.
    return_value (Any):Preset return value.
    checkf(Any):Command validation function with signature.
    no_displays(Any):Sensitive parameter names to filter.



Examples:
    >>> from lazyllm.common import LazyLLMCMD
    >>> cmd = LazyLLMCMD("run --epochs=50 --batch-size=32")
    >>> print(cmd.get_args("epochs"))
    50
    >>> print(cmd.get_args("batch-size")) 
    32
    >>> base = LazyLLMCMD("python train.py", checkf=lambda x: True)
    >>> new = base.with_cmd("python predict.py")
    
    """
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
                cmd = re.sub(pattern, '', cmd)
            return cmd
        else:
            return cmd

    def with_cmd(self, cmd):
        """Create new command object inheriting current configuration.

Args:
    cmd: New command content (must be same type as original)

"""
        # Attention: Cannot use copy.deepcopy because of class method.
        new_instance = LazyLLMCMD(cmd, return_value=self.return_value,
                                  checkf=self.checkf, no_displays=self.no_displays)
        return new_instance

    def get_args(self, key):
        """Extracts specified argument value from command string.

Args:
    key: Argument name
"""
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
    """
A lightweight read-only wrapper that holds an arbitrary object and exposes its attributes. It supports swapping the internal object dynamically and provides utility for checking emptiness. Note: it does not enforce deep immutability, but deepcopy drops the wrapped object.

Args:
    obj (Optional[Any]): The initial wrapped object, defaults to None.
"""
    def __init__(self, obj=None):
        self.obj = obj

    def set(self, obj):
        """
Replace the currently wrapped internal object.

Args:
    obj (Any): New object to wrap.
"""
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
        """
Check whether the wrapper currently holds no object.

Args:
    None.

**Returns:**

- bool: True if the internal object is None, otherwise False.
"""
        return self.obj is None


class Identity():
    """
Identity module that directly returns the input as output.

This module serves as a no-op placeholder in composition pipelines. If multiple inputs are provided, they are packed together before returning.

Args:
    *args: Optional positional arguments for placeholder compatibility.
    **kw: Optional keyword arguments for placeholder compatibility.
"""
    def __init__(self, *args, **kw):
        pass

    def __call__(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return package(*inputs)

    def __repr__(self):
        return make_repr('Module', 'Identity')


class ResultCollector(object):
    """A result collector used to store and access results by name during the execution of a flow or task.  
Calling the instance with a name returns a callable Impl object that collects results for that name.  
Useful for scenarios where intermediate results need to be shared across steps.
"""
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
    def keys(self):
        """Get all stored result names.

**Returns:**

- KeysView[str]: A set-like object containing result names.
"""
        return self._value.keys()
    def items(self):
        """Get all stored (name, value) pairs.

**Returns:**

- ItemsView[str, Any]: A set-like object containing name-value pairs of results.
"""
        return self._value.items()


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

def make_repr(category: str, type: str, *, name: Optional[str] = None,
              subs: Optional[List[str]] = None, attrs: Optional[Dict[str, Any]] = None, **kw):
    subs, attrs = subs or [], attrs or {}
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
    """Dynamic descriptor class for creating descriptors that support both instance and class level calls.

Args:
    func (callable): Function or method to be wrapped
"""
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
    """Environment variable context manager used to temporarily set environment variables during the execution of a code block, automatically restoring original environment variables upon exit.

Args:
    env_vars_dict (dict): Dictionary of environment variables to temporarily set; variables with None values are ignored.
"""
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

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_valid_path(path):
    return os.path.isfile(path)

class Finalizer(object):
    """Finalizer class for managing resource cleanup and release operations. Can be used as a context manager or trigger cleanup automatically when object is destroyed.

Args:
    func1 (Callable): Primary cleanup function. If func2 is provided, func1 is executed immediately and func2 becomes the cleanup function.
    func2 (Optional[Callable]): Optional cleanup function, defaults to None.
    condition (Callable): Condition function, cleanup is executed only when it returns True, defaults to always returning True.

Uses:
1. Can be used as a context manager (with statement)
2. Can trigger cleanup automatically when object is destroyed
3. Supports conditional cleanup
4. Supports two-phase initialization and cleanup

Note:
    - When func2 is provided, func1 is executed immediately during initialization
    - Cleanup function is executed only once
    - Cleanup occurs when object is destroyed or context is exited
"""
    def __init__(self, func1: Callable, func2: Optional[Callable] = None, *, condition: Callable = lambda: True):
        if func2:
            func1()
            func1 = func2
        self._func = func1
        self._condition = condition

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        if self._func:
            if self._condition(): self._func()
            self._func = None
