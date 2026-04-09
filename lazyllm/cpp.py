import importlib
import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

from lazyllm import config

config.add('cpp_switch', bool, False, 'ENABLE_CPP_OVERRIDE')
_LAZYLLM_CPP_MODULE = None


def _load_cpp_module():
    global _LAZYLLM_CPP_MODULE
    if _LAZYLLM_CPP_MODULE is None:
        _LAZYLLM_CPP_MODULE = importlib.import_module('lazyllm.lazyllm_cpp')
    return _LAZYLLM_CPP_MODULE


def _validate_method_aliases(value: Optional[Dict[str, str]]) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f'@cpp_proxy cpp_method_aliases must be dict[str, str], got: {type(value).__name__}'
        )
    for py_name, cpp_name in value.items():
        if not isinstance(py_name, str) or not isinstance(cpp_name, str):
            raise TypeError('@cpp_proxy cpp_method_aliases must be dict[str, str].')
    return value


def _build_valid_kwargs(impl_cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    'Extract params from kwargs, and kwargs only, for C++ object which name is the same and type is matched.'
    signature = inspect.signature(impl_cls.__init__)

    valid_params: Dict[str, Any] = {}
    for name, value in kwargs.items():
        param = signature.parameters.get(name)
        expected_type = param.annotation
        if type(value) is expected_type:
            valid_params[name] = value

    return valid_params


_C = TypeVar('_C', bound=type)

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


def _validate_name_list(value: Optional[List[str]], field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f'@cpp_proxy {field_name} must be list[str], got: {type(value).__name__}')
    if any(not isinstance(name, str) for name in value):
        raise TypeError(f'@cpp_proxy {field_name} must be list[str].')
    return value


def cpp_proxy(py_class: Optional[_C] = None):

    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_proxy can only decorate classes, got: {type(cls).__name__}')

        if not config.cpp_switch:
            return cls

        cpp_module = _load_cpp_module()
        default_impl_name = f'{cls.__name__}CPPImpl'
        if not hasattr(cpp_module, default_impl_name):
            raise AttributeError(f'@cpp_proxy cannot find C++ impl: {default_impl_name}')

        impl_holder = '_c_obj'
        impl_cls = getattr(cpp_module, default_impl_name)
        original_init = cls.__init__

        @wraps(original_init)
        def _proxied_init(self, *args, **kwargs):
            'Create C++ object instance right after Python __init__.'

            original_init(self, *args, **kwargs)

            valid_params = _build_valid_kwargs(impl_cls, kwargs)
            self.__setattr__(impl_holder, impl_cls(**valid_params))

        cls.__init__ = _proxied_init

        # Proxy C++ methods
        cpp_method_names: List[str] = []
        for name in dir(impl_cls):
            try:
                member = getattr(impl_cls, name)
            except Exception:
                continue
            if callable(member):
                cpp_method_names.append(name)

        if not proxy_names:
            auto_names = [name for name in cpp_method_names if hasattr(cls, name)]
            for py_name, cpp_name in method_aliases.items():
                if cpp_name in cpp_method_names and hasattr(cls, py_name) and py_name not in auto_names:
                    auto_names.append(py_name)
            proxy_method_names = auto_names
        else:
            proxy_method_names = list(proxy_names)

        def _make_method_proxy(py_name: str, cpp_name: str, original: Callable[..., Any]):
            @wraps(original)
            def _proxy(self, *args, **kwargs):
                # Keep subclass polymorphism on Python side unless that subclass
                # has its own @cpp_proxy decoration.
                if type(self) is not cls:
                    return original(self, *args, **kwargs)

                impl = getattr(self, impl_holder, None)
                if impl is None:
                    return original(self, *args, **kwargs)

                cpp_method = getattr(impl, cpp_name, None)
                if cpp_method is None:
                    return original(self, *args, **kwargs)

                try:
                    result = cpp_method(*args, **kwargs)
                except RuntimeError:
                    # Keep Python-side exception/warning semantics when
                    # C++ fast-path reports recoverable runtime errors.
                    return original(self, *args, **kwargs)
                return self if result is impl else result

            return _proxy

        for py_name in proxy_method_names:
            if not hasattr(cls, py_name):
                raise AttributeError(
                    f'@cpp_proxy funcs_to_override contains unknown method: {cls.__name__}.{py_name}'
                )

            cpp_name = method_aliases.get(py_name, py_name)
            if proxy_names and cpp_name not in cpp_method_names:
                raise AttributeError(
                    f'@cpp_proxy cannot find C++ method: {default_impl_name}.{cpp_name}'
                )

            original = getattr(cls, py_name)
            if not callable(original):
                continue
            setattr(cls, py_name, _make_method_proxy(py_name, cpp_name, original))

        if proxy_attrs:
            original_setattr = cls.__setattr__

            def _proxied_setattr(self, name, value):
                original_setattr(self, name, value)
                if name not in proxy_attrs:
                    return
                impl = getattr(self, impl_holder, None)
                if impl is None:
                    return
                try:
                    if hasattr(impl, name):
                        setattr(impl, name, value)
                except (AttributeError, TypeError, RuntimeError):
                    return

            cls.__setattr__ = _proxied_setattr

        return cls

    if py_class is None:
        return _decorate
    return _decorate(py_class)
