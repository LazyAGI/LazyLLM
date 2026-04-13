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
    '''Keep only kwargs whose names exist in impl __init__ and whose types match exactly.'''
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
    '''Instantiate the C++ object; if signature is unavailable, try valid kwargs subsets dynamically.'''
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

    # In some pybind build configurations, method signatures are not introspectable and
    # only "incompatible constructor arguments" is reported. In that case, try subsets
    # from larger to smaller to keep as many accepted kwargs as possible.
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
    '''Scan exported impl members and collect same-name methods/properties for proxying.'''
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
        if name not in py_cls.__dict__:
            continue

        py_member = py_cls.__dict__[name]
        if not callable(py_member):
            continue

        # Dynamic validation: if both signatures are available, require identical
        # parameter names; otherwise skip signature validation.
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


def cpp_class(py_class: Optional[_C] = None, *, cpp_class_name: Optional[str] = None):
    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_class can only decorate classes, got: {type(cls).__name__}')

        if not config.cpp_switch:
            return cls

        cpp_module = _load_cpp_module()
        export_name = cpp_class_name or cls.__name__
        cpp_export = getattr(cpp_module, export_name)
        return cast(_C, cpp_export)

    if py_class is None:
        return _decorate
    return _decorate(py_class)


def _resolve_impl_cls(cls: type[Any], cpp_class_name: Optional[str]) -> type:
    cpp_module = _load_cpp_module()
    impl_name = cpp_class_name or f'{cls.__name__}CPPImpl'
    if not hasattr(cpp_module, impl_name):
        raise AttributeError(f'@cpp_proxy cannot find C++ impl: {impl_name}')
    return getattr(cpp_module, impl_name)


def _install_proxied_init(cls: type[Any], impl_cls: type, impl_holder: str):
    original_init = cls.__init__

    @wraps(original_init)
    def _proxied_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        valid_params = _build_valid_kwargs(impl_cls, kwargs)
        impl = _instantiate_impl(impl_cls, valid_params)
        object.__setattr__(self, impl_holder, impl)

    cls.__init__ = _proxied_init


def _should_use_python_method(
    self,
    cls: type[Any],
    method_name: str,
    force_python_methods: set,
    fallback_rules: Dict[str, Tuple[str, ...]],
) -> bool:
    if type(self) is cls and method_name in force_python_methods:
        return True
    if type(self) is cls:
        return False

    deps = fallback_rules.get(method_name, ())
    return any(dep_name in type(self).__dict__ for dep_name in deps)


def _make_method_proxy(
    cls: type[Any],
    method_name: str,
    original_method,
    impl_holder: str,
    force_python_methods: set,
    fallback_rules: Dict[str, Tuple[str, ...]],
):
    @wraps(original_method)
    def _proxy(self, *args, **kwargs):
        if _should_use_python_method(self, cls, method_name, force_python_methods, fallback_rules):
            return original_method(self, *args, **kwargs)

        impl = getattr(self, impl_holder)
        cpp_method = getattr(impl, method_name)
        result = cpp_method(*args, **kwargs)
        return self if result is impl else result

    return _proxy


def _install_method_proxies(
    cls: type[Any],
    proxy_methods: Tuple[str, ...],
    impl_holder: str,
    force_python_methods: set,
    fallback_rules: Dict[str, Tuple[str, ...]],
):
    for method_name in proxy_methods:
        original_method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            _make_method_proxy(
                cls, method_name, original_method, impl_holder, force_python_methods, fallback_rules
            ),
        )


def _install_attr_proxy(cls: type[Any], proxy_attrs: Tuple[str, ...], impl_holder: str):
    if not proxy_attrs:
        return

    original_setattr = cls.__setattr__

    def _proxied_setattr(self, name, value):
        original_setattr(self, name, value)
        if name in proxy_attrs and hasattr(self, impl_holder):
            setattr(getattr(self, impl_holder), name, value)

    cls.__setattr__ = _proxied_setattr


def cpp_proxy(
    py_class: Optional[_C] = None,
    *,
    method_fallbacks: Optional[Dict[str, Tuple[str, ...]]] = None,
    python_methods_for_self: Tuple[str, ...] = (),
    cpp_class_name: Optional[str] = None,
):
    def _decorate(cls: _C) -> _C:
        if not isinstance(cls, type):
            raise TypeError(f'@cpp_proxy can only decorate classes, got: {type(cls).__name__}')

        if not config.cpp_switch:
            return cls

        impl_cls = _resolve_impl_cls(cls, cpp_class_name)
        proxy_methods, proxy_attrs = _scan_proxy_members(cls, impl_cls)
        fallback_rules = method_fallbacks or {}
        force_python_methods = set(python_methods_for_self)

        impl_holder = '_c_obj'
        _install_proxied_init(cls, impl_cls, impl_holder)
        _install_method_proxies(cls, proxy_methods, impl_holder, force_python_methods, fallback_rules)
        _install_attr_proxy(cls, proxy_attrs, impl_holder)

        return cls

    if py_class is None:
        return _decorate
    return _decorate(py_class)
