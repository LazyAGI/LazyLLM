import importlib.util
import pytest
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _reload_cpp_module():
    module_path = Path(__file__).resolve().parents[1] / 'lazyllm' / 'cpp.py'
    module_name = 'lazyllm.cpp'

    sys.modules.pop(module_name, None)
    sys.modules.pop('lazyllm', None)

    pkg = types.ModuleType('lazyllm')
    pkg.__path__ = [str(module_path.parent)]  # type: ignore[attr-defined]
    sys.modules['lazyllm'] = pkg

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    cpp = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = cpp
    spec.loader.exec_module(cpp)
    return cpp


def test_cpp_class_keeps_python_class_when_disabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '0')
    cpp = _reload_cpp_module()

    class PyOnly:
        pass

    replaced = cpp.cpp_class(PyOnly)
    assert replaced is PyOnly


def test_cpp_class_replaces_with_cpp_export_when_enabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    class CppDummy:
        pass

    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(Dummy=CppDummy))

    class Dummy:
        pass

    replaced = cpp.cpp_class(Dummy)
    assert replaced is CppDummy


def test_cpp_class_rejects_non_class_object(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    with pytest.raises(TypeError, match='can only decorate classes'):
        cpp.cpp_class('NotAClass')


def test_cpp_class_raises_when_cpp_export_missing(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()
    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace())

    class Missing:
        pass

    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        cpp.cpp_class(Missing)


def test_cpp_class_propagates_import_error_when_enabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    def _boom():
        raise ImportError('boom')

    monkeypatch.setattr(cpp, '_load_cpp_module', _boom)

    class AnyClass:
        pass

    with pytest.raises(ImportError, match='boom'):
        cpp.cpp_class(AnyClass)


def test_cpp_class_overrides_selected_methods_when_enabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    class CppDummy:
        pass

    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(Dummy=CppDummy))

    class Dummy:
        def keep(self):
            return 'py'

        def drop(self):
            return 'cpp'

    replaced = cpp.cpp_class(funcs_to_override=['keep'])(Dummy)
    assert replaced is CppDummy
    assert replaced.keep is Dummy.keep
    assert not hasattr(replaced, 'drop')


def test_cpp_class_overrides_dunder_init_when_enabled(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    class CppDummy:
        def __init__(self):
            self.value = 'cpp'

    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(Dummy=CppDummy))

    class Dummy:
        def __init__(self, value):
            self.value = value

    replaced = cpp.cpp_class(funcs_to_override=['__init__'])(Dummy)
    inst = replaced('py')
    assert inst.value == 'py'


def test_cpp_class_rejects_invalid_funcs_to_override(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()

    with pytest.raises(TypeError, match='funcs_to_override must be list\\[str\\]'):
        cpp.cpp_class(funcs_to_override='keep')

    with pytest.raises(TypeError, match='funcs_to_override must be list\\[str\\]'):
        cpp.cpp_class(funcs_to_override=['ok', 1])


def test_cpp_class_raises_for_unknown_override_method(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', '1')
    cpp = _reload_cpp_module()
    monkeypatch.setattr(cpp, '_load_cpp_module', lambda: SimpleNamespace(Dummy=type('CppDummy', (), {})))

    class Dummy:
        pass

    with pytest.raises(AttributeError, match='unknown method'):
        cpp.cpp_class(funcs_to_override=['missing'])(Dummy)
