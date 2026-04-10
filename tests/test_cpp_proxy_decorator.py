import importlib.util
import os
import pytest
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class _ConfigStub:
    def __init__(self):
        self._values = {'home': str(Path.home() / '.lazyllm')}

    def add(self, name, _type, default, env_name, *args, **kwargs):
        raw = os.getenv(f'LAZYLLM_{env_name}')
        if raw is None:
            value = default
        elif _type is bool:
            value = raw.lower() in {'1', 'true', 'yes', 'on'}
        else:
            value = _type(raw)
        self._values[name] = value
        setattr(self, name, value)

    def __getitem__(self, key):
        return self._values[key]


def _reload_cpp_module():
    module_path = Path(__file__).resolve().parents[1] / 'lazyllm' / 'cpp.py'
    module_name = 'lazyllm.cpp'

    sys.modules.pop(module_name, None)
    sys.modules.pop('lazyllm', None)

    pkg = types.ModuleType('lazyllm')
    pkg.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]
    pkg.config = _ConfigStub()  # type: ignore[attr-defined]
    sys.modules['lazyllm'] = pkg

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    cpp = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = cpp
    spec.loader.exec_module(cpp)
    return cpp


class _DemoCPPImpl:
    __proxy_methods__ = ('foo',)
    __proxy_method_signatures__ = {'foo': ('x',)}
    __proxy_attrs__ = ('value',)
    __init_param_types__ = {'count': int}

    def __init__(self, count: int = 0):
        self.count = count
        self.value = 0

    def foo(self, x):
        return x + self.count


def test_cpp_proxy_keeps_python_class_when_disabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '0')
    cpp = _reload_cpp_module()

    class Demo:
        pass

    proxied = cpp.cpp_proxy(Demo)
    assert proxied is Demo


def test_cpp_proxy_proxies_method_and_attr(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()
    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(DemoCPPImpl=_DemoCPPImpl))

    @cpp.cpp_proxy
    class Demo:
        def __init__(self, count: int = 0, ignore: str = ''):
            self.value = 3

        def foo(self, x):
            return -1

    obj = Demo(count=2, ignore='x')
    assert obj.foo(5) == 7
    assert obj._c_obj.count == 2

    obj.value = 9
    assert obj._c_obj.value == 9


def test_cpp_proxy_filters_kwargs_by_name_and_type(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()
    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(DemoCPPImpl=_DemoCPPImpl))

    @cpp.cpp_proxy
    class Demo:
        def __init__(self, count: int = 0, ignore: str = ''):
            self.value = 0

        def foo(self, x):
            return -1

    obj = Demo(count='2', ignore='x')
    # count 类型不匹配 int，不会透传到 C++ 构造函数。
    assert obj._c_obj.count == 0


def test_cpp_proxy_raises_on_method_signature_mismatch(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    class ImplWithMismatch(_DemoCPPImpl):
        __proxy_method_signatures__ = {'foo': ('x', 'y')}

    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(DemoCPPImpl=ImplWithMismatch))

    with pytest.raises(TypeError, match='Signature mismatch'):
        @cpp.cpp_proxy
        class Demo:
            def __init__(self, count: int = 0):
                self.value = 0

            def foo(self, x):
                return -1


def test_cpp_proxy_raises_when_python_method_missing(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()
    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(DemoCPPImpl=_DemoCPPImpl))

    with pytest.raises(AttributeError, match='missing method for proxy'):
        @cpp.cpp_proxy
        class Demo:
            def __init__(self, count: int = 0):
                self.value = 0
