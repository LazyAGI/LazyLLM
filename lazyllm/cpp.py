import importlib
import inspect
from functools import wraps
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


def _build_valid_kwargs(init_param_types: Dict[str, type], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    valid_params: Dict[str, Any] = {}
    for name, value in kwargs.items():
        expected_type = init_param_types.get(name)
        if expected_type is None:
            continue
        if type(value) is expected_type:
            valid_params[name] = value
    return valid_params


def _validate_proxy_contract(py_cls: type, impl_cls: type):
    proxy_methods = getattr(impl_cls, '__proxy_methods__', None)
    method_signatures = getattr(impl_cls, '__proxy_method_signatures__', None)
    proxy_attrs = getattr(impl_cls, '__proxy_attrs__', None)
    init_param_types = getattr(impl_cls, '__init_param_types__', None)

    if not isinstance(proxy_methods, (tuple, list)) or any(not isinstance(name, str) for name in proxy_methods):
        raise TypeError(f'{impl_cls.__name__}.__proxy_methods__ must be tuple/list[str].')
    if not isinstance(proxy_attrs, (tuple, list)) or any(not isinstance(name, str) for name in proxy_attrs):
        raise TypeError(f'{impl_cls.__name__}.__proxy_attrs__ must be tuple/list[str].')
    if not isinstance(method_signatures, dict):
        raise TypeError(f'{impl_cls.__name__}.__proxy_method_signatures__ must be dict[str, tuple[str, ...]].')
    if set(method_signatures.keys()) != set(proxy_methods):
        raise TypeError(f'{impl_cls.__name__}.__proxy_method_signatures__ keys must equal __proxy_methods__.')
    if not isinstance(init_param_types, dict):
        raise TypeError(f'{impl_cls.__name__}.__init_param_types__ must be dict[str, type].')

    for name, expected_sig in method_signatures.items():
        if not isinstance(expected_sig, (tuple, list)) or any(not isinstance(p, str) for p in expected_sig):
            raise TypeError(
                f'{impl_cls.__name__}.__proxy_method_signatures__["{name}"] must be tuple/list[str].'
            )

    for name, expected_type in init_param_types.items():
        if not isinstance(name, str) or not isinstance(expected_type, type):
            raise TypeError(f'{impl_cls.__name__}.__init_param_types__ must be dict[str, type].')

    for name in proxy_methods:
        if not hasattr(impl_cls, name):
            raise AttributeError(f'{impl_cls.__name__} missing exported proxy method: {name}')
        impl_member = getattr(impl_cls, name)
        if not callable(impl_member):
            raise TypeError(f'{impl_cls.__name__}.{name} must be callable')

        if not hasattr(py_cls, name):
            raise AttributeError(f'{py_cls.__name__} missing method for proxy: {name}')
        py_member = getattr(py_cls, name)
        if not callable(py_member):
            raise TypeError(f'{py_cls.__name__}.{name} must be callable')

        expected_sig = tuple(method_signatures[name])
        py_sig = _normalize_param_names(py_member)
        if py_sig != expected_sig:
            raise TypeError(
                f'Signature mismatch for {py_cls.__name__}.{name}: '
                f'python params={py_sig}, expected={expected_sig}'
            )

    return tuple(proxy_methods), tuple(proxy_attrs), dict(init_param_types)


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


def cpp_proxy(py_class: Optional[_C]):
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
        proxy_methods, proxy_attrs, init_param_types = _validate_proxy_contract(cls, impl_cls)

        impl_holder = '_c_obj'
        original_init = cls.__init__

        @wraps(original_init)
        def _proxied_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            valid_params = _build_valid_kwargs(init_param_types, kwargs)
            impl = impl_cls(**valid_params)
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
