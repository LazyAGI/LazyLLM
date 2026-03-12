import importlib
import pytest
from types import SimpleNamespace


def _reload_cpp_module():
    import lazyllm.cpp as cpp
    return importlib.reload(cpp)


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
