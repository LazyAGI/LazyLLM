import copy
import builtins
import itertools
from typing import Callable, Any, Optional, List
from .globals import globals
from .common import package


class AttrTree(object):
    def __init__(self, name: Optional[str] = None, pres: Optional[List[str]] = None):
        self._path = copy.deepcopy(pres or [])
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

    def __reduce__(self) -> tuple[Any, ...]:
        return Placeholder.__new__, (Placeholder, self.idx)

for i in range(10):
    vars()[f'_{i}'] = Placeholder(i)

def _setattr(self, key, v):
    raise RuntimeError(f'Cannot set attr for Placeholder, you want to set {key}={v}')
Placeholder.__setattr__ = _setattr


class _MetaBind(type):
    def __instancecheck__(self, __instance):
        if isinstance(__instance, Bind) and isinstance(__instance._f, self):
            return True
        return super(__class__, self).__instancecheck__(__instance)


class Bind(object):
    """The Bind class provides function binding and deferred invocation capabilities, supporting dynamic argument passing and context-based argument resolution for flexible function composition and pipeline-style calls.

The bind function binds a callable with fixed positional and keyword arguments, supporting placeholders (e.g. _0, _1) to reference outputs of upstream nodes within the current pipeline, enabling flexible data jumps and function composition.

Notes:
    - Bound arguments can be concrete values or placeholders referring to upstream pipeline outputs.
    - Bindings are local to the current pipeline context and do not support cross-pipeline or external variable binding.

Args:
    __bind_func (Callable or type): The function or function type to bind. If a type is given, it will be instantiated automatically.
    *args: Fixed positional arguments to bind, supporting placeholders.
    **kw: Fixed keyword arguments to bind, supporting placeholders.


Examples:
    >>> from lazyllm import bind, _0, _1
    >>> def f1(x):
    ...     return x ** 2
    >>> def f21(input1, input2=0):
    ...     return input1 + input2 + 1
    >>> def f22(input1, input2=0):
    ...     return input1 + input2 - 1
    >>> def f3(in1='placeholder1', in2='placeholder2', in3='placeholder3'):
    ...     return f"get [input:{in1}], [f21:{in2}], [f22:{in3}]"
    
    >>> from lazyllm import pipeline, parallel
    
    >>> with pipeline() as ppl:
    ...     ppl.f1 = f1
    ...     with parallel() as ppl.subprl2:
    ...         ppl.subprl2.path1 = f21
    ...         ppl.subprl2.path2 = f22
    ...     ppl.f3 = bind(f3, ppl.input, _0, _1)
    ...
    >>> print(ppl(2))
    get [input:2], [f21:5], [f22:3]
    
    >>> # Demonstrate operator '|' overloading for bind
    >>> with pipeline() as ppl2:
    ...     ppl2.f1 = f1
    ...     with parallel().bind(ppl2.input, _0) as ppl2.subprl2:
    ...         ppl2.subprl2.path1 = f21
    ...         ppl2.subprl2.path2 = f22
    ...     ppl2.f3 = f3 | bind(ppl2.input, _0, _1)
    ...
    >>> print(ppl2(2))
    get [input:2], [f21:7], [f22:5]
    """
    class _None: pass

    class Args(object):
        class _None: pass
        class Unpack(package): pass

        def __init__(self, source_id: str, target_id: str = 'input', *, unpack: bool = False):
            self._item_key, self._attr_key = Bind.Args._None, Bind.Args._None
            self._source_id, self._target_id = source_id, target_id
            self._unpack = unpack

        def __getitem__(self, key: str):
            self._item_key = key
            return self

        def __getattr__(self, key: str):
            if key.startswith('__') and key.endswith('__'):
                raise AttributeError(f'Args has no attribute {key}')
            self._attr_key = key
            return self

        def __getstate__(self):
            return self._item_key, self._attr_key, self._source_id, self._target_id

        def __setstate__(self, state):
            self._item_key, self._attr_key, self._source_id, self._target_id = state

        def get_arg(self, source):
            if (not source or self._source_id != source['source']) and self._source_id in globals['bind_args']:
                source = globals['bind_args'][self._source_id]
            if not source or source['source'] != self._source_id:
                raise RuntimeError('Unable to find the bound parameter, possibly due to pipeline.input/output can only '
                                   'be bind in direct member of pipeline! You may solve this by defining the pipeline '
                                   'in a `with lazyllm.save_pipeline_result():` block.')
            input = result = source[self._target_id]
            source = source['source']
            if self._item_key is not Bind.Args._None: result = input[self._item_key]
            elif self._attr_key is not Bind.Args._None: result = getattr(input, self._attr_key)
            if self._unpack and isinstance(result, package): result = Bind.Args.Unpack(result)
            return result

    def __init__(self, __bind_func=_None, *args, **kw):
        self._f = __bind_func() if isinstance(__bind_func, type) and __bind_func is not Bind._None else __bind_func
        self._args = args
        self._kw = kw
        self._has_root = (any([isinstance(a, AttrTree) for a in args])
                          or any([isinstance(v, AttrTree) for v in kw.values()]))

    def __ror__(self, __value: Callable):
        if self._f is not Bind._None: self._args = (self._f,) + self._args
        self._f = __value
        return self

    # _bind_args_source: dict(input=input, args=dict(key=value))
    def __call__(self, *args, _bind_args_source=None, **kw):
        if self._f is None: return None
        keys = set(kw.keys()).intersection(set(self._kw.keys()))
        assert len(keys) == 0, f'Keys `{keys}` are already bind!'
        bind_args = args if len(self._args) == 0 else (
            [args[a.idx] if isinstance(a, Placeholder) else a for a in self._args])
        kwargs = {k: args[v.idx] if isinstance(v, Placeholder) else v for k, v in self._kw.items()}
        bind_args = [a.get_arg(_bind_args_source) if isinstance(a, Bind.Args) else a for a in bind_args]
        bind_args = list(itertools.chain.from_iterable(x if isinstance(x, Bind.Args.Unpack) else [x] for x in bind_args))
        kwargs = {k: v.get_arg(_bind_args_source) if isinstance(v, Bind.Args) else v for k, v in kwargs.items()}
        return self._f(*bind_args, **kwargs, **kw)

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


builtins.bind = Bind
