import importlib
import inspect
import re
from functools import wraps
from itertools import combinations
from typing import Any, Dict, Optional, Tuple, TypeVar, cast

from lazyllm import config

config.add('cpp_switch', bool, False, 'ENABLE_CPP_OVERRIDE')

_LAZYLLM_CPP_MODULE = None
_C = TypeVar('_C', bound=type)


def _load_cpp_module():
    global _LAZYLLM_CPP_MODULE
    if _LAZYLLM_CPP_MODULE is None:
        _LAZYLLM_CPP_MODULE = importlib.import_module('lazyllm.lazyllm_cpp')
    return _LAZYLLM_CPP_MODULE


def _normalize_param_names(callable_obj: Any) -> Tuple[str, ...]:
    signature = inspect.signature(callable_obj)
    names = []
    for index, param in enumerate(signature.parameters.values()):
        if index == 0 and param.name in ('self', 'cls'):
            continue
        names.append(param.name)
    return tuple(names)


def _build_valid_kwargs(impl_cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """只保留与 impl __init__ 同名且类型严格匹配的 kwargs。"""
    try:
        signature = inspect.signature(impl_cls.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)

    valid_params: Dict[str, Any] = {}
    for name, value in kwargs.items():
        param = signature.parameters.get(name)
        if param is None:
            continue

        expected_type = param.annotation
        if expected_type is inspect._empty or not isinstance(expected_type, type):
            continue

        if type(value) is expected_type:
            valid_params[name] = value

    return valid_params


def _instantiate_impl(impl_cls: type, kwargs: Dict[str, Any]):
    """实例化 C++ 对象；若签名不可见，则动态尝试可行 kwargs 子集。"""
    candidate_kwargs = dict(kwargs)
    while True:
        try:
            return impl_cls(**candidate_kwargs)
        except TypeError as exc:
            message = str(exc)
            match = re.search(r"unexpected keyword argument '([^']+)'", message)
            if not match:
                break
            bad_name = match.group(1)
            if bad_name not in candidate_kwargs:
                break
            candidate_kwargs.pop(bad_name)

    # pybind 方法在某些构建配置下没有可见签名，报错只会是“incompatible constructor arguments”。
    # 这种情况下按“参数个数从多到少”尝试子集，保留尽可能多的可接受参数。
    last_exc = None
    keys = tuple(kwargs.keys())
    for size in range(len(keys), -1, -1):
        for subset in combinations(keys, size):
            subset_kwargs = {k: kwargs[k] for k in subset}
            try:
                return impl_cls(**subset_kwargs)
            except TypeError as exc:
                last_exc = exc
                continue

    if last_exc is not None:
        raise last_exc
    raise TypeError(f'Failed to construct {impl_cls.__name__} with kwargs: {kwargs}')


def _scan_proxy_members(py_cls: type, impl_cls: type):
    """扫描 impl 导出成员，收集可代理的同名方法和属性。"""
    proxy_methods = []
    proxy_attrs = []

    for name, member in impl_cls.__dict__.items():
        if name.startswith('__'):
            continue

        if isinstance(member, property):
            proxy_attrs.append(name)
            continue

        if not callable(member):
            continue
        if not hasattr(py_cls, name):
            continue

        py_member = getattr(py_cls, name)
        if not callable(py_member):
            continue

        # 动态校验：若能拿到双方签名，则要求参数名一致；拿不到则跳过签名校验。
        try:
            py_sig = _normalize_param_names(py_member)
        except (TypeError, ValueError):
            py_sig = None
        try:
            impl_sig = _normalize_param_names(member)
        except (TypeError, ValueError):
            impl_sig = None
        if py_sig is not None and impl_sig is not None and py_sig != impl_sig:
            raise TypeError(
                f'Signature mismatch for {py_cls.__name__}.{name}: '
                f'python params={py_sig}, cpp params={impl_sig}'
            )

        proxy_methods.append(name)

    return tuple(proxy_methods), tuple(proxy_attrs)


def cpp_class(py_class: Optional[_C] = None):
    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_class can only decorate classes, got: {type(cls).__name__}')

        if not config.cpp_switch:
            return cls

        cpp_module = _load_cpp_module()
        export_name = cls.__name__
        cpp_export = getattr(cpp_module, export_name)
        return cast(_C, cpp_export)

    if py_class is None:
        return _decorate
    return _decorate(py_class)


def cpp_proxy(py_class: Optional[_C] = None):
    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_proxy can only decorate classes, got: {type(cls).__name__}')

        if not config.cpp_switch:
            return cls

        cpp_module = _load_cpp_module()
        impl_name = f'{cls.__name__}CPPImpl'
        if not hasattr(cpp_module, impl_name):
            raise AttributeError(f'@cpp_proxy cannot find C++ impl: {impl_name}')

        impl_cls = getattr(cpp_module, impl_name)
        proxy_methods, proxy_attrs = _scan_proxy_members(cls, impl_cls)

        impl_holder = '_c_obj'
        original_init = cls.__init__

        @wraps(original_init)
        def _proxied_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            valid_params = _build_valid_kwargs(impl_cls, kwargs)
            impl = _instantiate_impl(impl_cls, valid_params)
            object.__setattr__(self, impl_holder, impl)

        cls.__init__ = _proxied_init

        def _make_method_proxy(method_name: str, original_method):
            @wraps(original_method)
            def _proxy(self, *args, **kwargs):
                impl = getattr(self, impl_holder)
                cpp_method = getattr(impl, method_name)
                result = cpp_method(*args, **kwargs)
                return self if result is impl else result

            return _proxy

        for method_name in proxy_methods:
            original_method = getattr(cls, method_name)
            setattr(cls, method_name, _make_method_proxy(method_name, original_method))

        if proxy_attrs:
            original_setattr = cls.__setattr__

            def _proxied_setattr(self, name, value):
                original_setattr(self, name, value)
                if name in proxy_attrs and hasattr(self, impl_holder):
                    setattr(getattr(self, impl_holder), name, value)

            cls.__setattr__ = _proxied_setattr

        return cls

    if py_class is None:
        return _decorate
    return _decorate(py_class)
