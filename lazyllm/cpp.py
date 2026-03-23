import importlib
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

_LAZYLLM_CPP_MODULE = None
_LAZYLLM_CPP_ENABLED = None
_C = TypeVar('_C', bound=type)
_CLASS_ATTR_MISSING = object()


def _is_enabled() -> bool:
    global _LAZYLLM_CPP_ENABLED
    if _LAZYLLM_CPP_ENABLED is None:
        value = os.getenv('LAZYLLM_ENABLE_CPP_OVERRIDE')
        _LAZYLLM_CPP_ENABLED = value is not None and (value == '1' or value.lower() == 'true')
    return _LAZYLLM_CPP_ENABLED


def _load_cpp_module():
    global _LAZYLLM_CPP_MODULE
    if _LAZYLLM_CPP_MODULE is None:
        _LAZYLLM_CPP_MODULE = importlib.import_module('lazyllm.lazyllm_cpp')
    return _LAZYLLM_CPP_MODULE


def _validate_funcs_to_override(funcs_to_override: Optional[List[str]]) -> List[str]:
    if funcs_to_override is None:
        return []
    if not isinstance(funcs_to_override, list):
        raise TypeError(
            f'@cpp_class funcs_to_override must be list[str], got: {type(funcs_to_override).__name__}'
        )
    invalid = [name for name in funcs_to_override if not isinstance(name, str)]
    if invalid:
        raise TypeError('@cpp_class funcs_to_override must be list[str].')
    return funcs_to_override


def _validate_attrs_to_proxy(attrs_to_proxy: Optional[List[str]]) -> List[str]:
    if attrs_to_proxy is None:
        return []
    if not isinstance(attrs_to_proxy, list):
        raise TypeError(
            f'@cpp_proxy attrs_to_proxy must be list[str], got: {type(attrs_to_proxy).__name__}'
        )
    invalid = [name for name in attrs_to_proxy if not isinstance(name, str)]
    if invalid:
        raise TypeError('@cpp_proxy attrs_to_proxy must be list[str].')
    return attrs_to_proxy


def _validate_method_aliases(method_aliases: Optional[Dict[str, str]]) -> Dict[str, str]:
    if method_aliases is None:
        return {}
    if not isinstance(method_aliases, dict):
        raise TypeError(
            f'@cpp_proxy cpp_method_aliases must be dict[str, str], got: {type(method_aliases).__name__}'
        )
    for py_name, cpp_name in method_aliases.items():
        if not isinstance(py_name, str) or not isinstance(cpp_name, str):
            raise TypeError('@cpp_proxy cpp_method_aliases must be dict[str, str].')
    return method_aliases


def _lookup_class_attr(runtime_cls: type, name: str):
    for base in runtime_cls.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return _CLASS_ATTR_MISSING


def _is_data_attr_on_impl(impl: Any, name: str) -> bool:
    try:
        value = getattr(impl, name)
    except (AttributeError, RuntimeError):
        return False
    return not callable(value)


def _method_overridden(runtime_cls: type, base_cls: type, method_name: str) -> bool:
    runtime_method = getattr(runtime_cls, method_name, None)
    base_method = getattr(base_cls, method_name, None)
    return runtime_method is not base_method


def _resolve_impl_cls(cpp_module: Any, runtime_cls: type, default_impl_name: str):
    default_impl_cls = getattr(cpp_module, default_impl_name)
    impl_name = getattr(runtime_cls, '__cpp_proxy_impl_name__', f'{runtime_cls.__name__}CPPImpl')
    return getattr(cpp_module, impl_name, default_impl_cls)


def _create_impl(
    cpp_module: Any,
    runtime_cls: type,
    default_impl_name: str,
    init_state: Dict[str, Any],
    init_args: Tuple[Any, ...],
    init_kwargs: Dict[str, Any],
):
    impl_cls = _resolve_impl_cls(cpp_module, runtime_cls, default_impl_name)

    attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    attempts.append((init_args, dict(init_kwargs)))
    attempts.append(((), {}))

    last_error: Optional[Exception] = None
    impl = None
    for args, kwargs in attempts:
        try:
            impl = impl_cls(*args, **kwargs)
            break
        except TypeError as exc:
            last_error = exc

    if impl is not None:
        for name, value in init_state.items():
            if not hasattr(impl, name):
                continue
            try:
                setattr(impl, name, value)
            except (AttributeError, TypeError):
                continue
        return impl

    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Failed to initialize C++ proxy impl for {runtime_cls.__name__}')


def cpp_class(py_class: Optional[_C] = None, *, funcs_to_override: Optional[List[str]] = None):
    override_names = _validate_funcs_to_override(funcs_to_override)

    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_class can only decorate classes, got: {type(cls).__name__}')

        if not _is_enabled():
            return cls

        cpp_module = _load_cpp_module()
        export_name = cls.__name__

        cpp_export = getattr(cpp_module, export_name)
        for name in override_names:
            if not hasattr(cls, name):
                raise AttributeError(
                    f'@cpp_class funcs_to_override contains unknown method: {cls.__name__}.{name}'
                )
            method = cls.__dict__.get(name, getattr(cls, name))
            setattr(cpp_export, name, method)
        return cast(_C, cpp_export)

    if py_class is None:
        return _decorate
    return _decorate(py_class)


def cpp_proxy(  # noqa: C901
    py_class: Optional[_C] = None,
    *,
    funcs_to_override: Optional[List[str]] = None,
    attrs_to_proxy: Optional[List[str]] = None,
    cpp_method_aliases: Optional[Dict[str, str]] = None,
):
    proxy_names = _validate_funcs_to_override(funcs_to_override)
    proxy_attrs = set(_validate_attrs_to_proxy(attrs_to_proxy))
    method_aliases = _validate_method_aliases(cpp_method_aliases)

    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_proxy can only decorate classes, got: {type(cls).__name__}')

        if not _is_enabled():
            return cls

        cpp_module = _load_cpp_module()
        default_impl_name = f'{cls.__name__}CPPImpl'
        if not hasattr(cpp_module, default_impl_name):
            raise AttributeError(f'@cpp_proxy cannot find C++ impl: {default_impl_name}')

        cls.__cpp_proxy_impl_name__ = default_impl_name

        original_init = cls.__init__
        original_getattribute = cls.__getattribute__
        original_setattr = cls.__setattr__
        impl_holder = '_cpp_impl'

        @wraps(original_init)
        def _proxied_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            existing = object.__getattribute__(self, '__dict__').get(impl_holder)
            if existing is not None:
                return
            state = dict(object.__getattribute__(self, '__dict__'))
            impl = _create_impl(cpp_module, type(self), default_impl_name, state, args, kwargs)
            object.__setattr__(self, impl_holder, impl)
            shadow_state = object.__getattribute__(self, '__dict__')
            for attr_name in list(shadow_state.keys()):
                if attr_name == impl_holder:
                    continue
                if hasattr(impl, attr_name):
                    shadow_state.pop(attr_name, None)

        cls.__init__ = _proxied_init

        def _get_impl(self):
            return object.__getattribute__(self, '__dict__').get(impl_holder)

        def _make_method_proxy(name: str, original: Callable[..., Any]):
            target_name = method_aliases.get(name, name)

            @wraps(original)
            def _proxy(self, *args, **kwargs):
                impl = _get_impl(self)
                if impl is None:
                    return original(self, *args, **kwargs)
                if name in {'split_text', '_merge'}:
                    runtime_cls = type(self)
                    # If a subclass customizes split/merge hooks, keep Python behavior
                    # so polymorphism still follows Python overrides.
                    if (
                        _method_overridden(runtime_cls, cls, '_split')
                        or _method_overridden(runtime_cls, cls, '_merge')
                    ):
                        return original(self, *args, **kwargs)
                try:
                    result = getattr(impl, target_name)(*args, **kwargs)
                except RuntimeError as exc:
                    if name == 'split_text':
                        msg = str(exc)
                        if 'close to chunk size' in msg:
                            # Keep Python warning behavior for tiny effective chunk sizes.
                            return original(self, *args, **kwargs)
                        if 'longer than chunk size' in msg:
                            raise ValueError(msg) from exc
                    raise
                return self if result is impl else result
            return _proxy

        for name in proxy_names:
            if not hasattr(cls, name):
                raise AttributeError(
                    f'@cpp_proxy funcs_to_override contains unknown method: {cls.__name__}.{name}'
                )
            original = getattr(cls, name)
            setattr(cls, name, _make_method_proxy(name, original))

        def _proxied_getattribute(self, name):
            if name == impl_holder:
                return original_getattribute(self, name)
            impl = object.__getattribute__(self, '__dict__').get(impl_holder)
            if impl is not None:
                cls_attr = _lookup_class_attr(type(self), name)
                if name in proxy_attrs or cls_attr is _CLASS_ATTR_MISSING:
                    if _is_data_attr_on_impl(impl, name):
                        return getattr(impl, name)
            return original_getattribute(self, name)

        def _proxied_setattr(self, name, value):
            if name == impl_holder:
                original_setattr(self, name, value)
                return
            impl = object.__getattribute__(self, '__dict__').get(impl_holder)
            if impl is not None:
                cls_attr = _lookup_class_attr(type(self), name)
                if (name in proxy_attrs or cls_attr is _CLASS_ATTR_MISSING) and hasattr(impl, name):
                    setattr(impl, name, value)
                    object.__getattribute__(self, '__dict__').pop(name, None)
                    return
            original_setattr(self, name, value)

        cls.__getattribute__ = _proxied_getattribute
        cls.__setattr__ = _proxied_setattr

        return cls

    if py_class is None:
        return _decorate
    return _decorate(py_class)
