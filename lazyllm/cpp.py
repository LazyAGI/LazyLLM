import importlib
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

from lazyllm import config

config.add('cpp_switch', bool, False, 'ENABLE_CPP_OVERRIDE')

_LAZYLLM_CPP_MODULE = None
_C = TypeVar('_C', bound=type)


def _load_cpp_module():
    global _LAZYLLM_CPP_MODULE
    if _LAZYLLM_CPP_MODULE is None:
        _LAZYLLM_CPP_MODULE = importlib.import_module('lazyllm.lazyllm_cpp')
    return _LAZYLLM_CPP_MODULE


def _validate_name_list(value: Optional[List[str]], field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f'@cpp_proxy {field_name} must be list[str], got: {type(value).__name__}')
    if any(not isinstance(name, str) for name in value):
        raise TypeError(f'@cpp_proxy {field_name} must be list[str].')
    return value


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


def _resolve_impl_cls(cpp_module: Any, runtime_cls: type, default_impl_name: str):
    default_impl = getattr(cpp_module, default_impl_name)
    impl_name = getattr(runtime_cls, '__cpp_proxy_impl_name__', default_impl_name)
    return getattr(cpp_module, impl_name, default_impl)


def _create_impl_instance(
    impl_cls: type,
    init_args: Tuple[Any, ...],
    init_kwargs: Dict[str, Any],
):
    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
        (init_args, dict(init_kwargs)),
        ((), {}),
    ]
    last_error: Optional[TypeError] = None
    for args, kwargs in attempts:
        try:
            return impl_cls(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Failed to initialize C++ proxy impl: {impl_cls.__name__}')


def _sync_state_to_impl(impl: Any, state: Dict[str, Any], impl_holder_name: str) -> None:
    for attr_name, value in state.items():
        if attr_name == impl_holder_name:
            continue
        try:
            if hasattr(impl, attr_name):
                setattr(impl, attr_name, value)
        except (AttributeError, TypeError, RuntimeError):
            continue


def _public_callable_names(cls: type) -> List[str]:
    names: List[str] = []
    for name in dir(cls):
        if name.startswith('_'):
            continue
        try:
            member = getattr(cls, name)
        except Exception:
            continue
        if callable(member):
            names.append(name)
    return names


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


def cpp_proxy(
    py_class: Optional[_C] = None,
    *,
    funcs_to_override: Optional[List[str]] = None,
    attrs_to_proxy: Optional[List[str]] = None,
    cpp_method_aliases: Optional[Dict[str, str]] = None,
):
    proxy_names = _validate_name_list(funcs_to_override, 'funcs_to_override')
    proxy_attrs = set(_validate_name_list(attrs_to_proxy, 'attrs_to_proxy'))
    method_aliases = _validate_method_aliases(cpp_method_aliases)

    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_proxy can only decorate classes, got: {type(cls).__name__}')

        if not config.cpp_switch:
            return cls

        cpp_module = _load_cpp_module()
        default_impl_name = f'{cls.__name__}CPPImpl'
        if not hasattr(cpp_module, default_impl_name):
            raise AttributeError(f'@cpp_proxy cannot find C++ impl: {default_impl_name}')

        cls.__cpp_proxy_impl_name__ = default_impl_name

        impl_holder = '_c_obj'
        original_init = cls.__init__

        @wraps(original_init)
        def _proxied_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if getattr(self, impl_holder, None) is not None:
                return

            impl_cls = _resolve_impl_cls(cpp_module, type(self), default_impl_name)
            impl = _create_impl_instance(impl_cls, args, kwargs)
            object.__setattr__(self, impl_holder, impl)
            _sync_state_to_impl(impl, dict(getattr(self, '__dict__', {})), impl_holder)

        cls.__init__ = _proxied_init

        cpp_method_names = set(_public_callable_names(getattr(cpp_module, default_impl_name)))

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
