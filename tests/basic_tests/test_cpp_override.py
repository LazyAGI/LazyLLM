import sys
import types

import lazyllm.cpp as cpp


def test_cpp_override_disabled(monkeypatch):
    monkeypatch.delenv('LAZYLLM_ENABLE_CPP_OVERRIDE', raising=False)

    fake_globals = {'__name__': 'fake.module', 'SentenceSplitter': object()}
    assert cpp.override_with_cpp_exports(fake_globals, ('SentenceSplitter',)) == []


def test_cpp_override_applies_by_same_name(monkeypatch):
    class PySentenceSplitter:
        pass

    class CppSentenceSplitter:
        pass

    cpp_module = types.ModuleType('lazyllm.lazyllm_cpp')
    cpp_module.SentenceSplitter = CppSentenceSplitter

    monkeypatch.setitem(sys.modules, 'lazyllm.lazyllm_cpp', cpp_module)
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', 'ON')

    fake_globals = {'__name__': 'fake.module', 'SentenceSplitter': PySentenceSplitter}
    applied = cpp.override_with_cpp_exports(fake_globals, ('SentenceSplitter',))

    assert applied == ['SentenceSplitter']
    assert fake_globals['SentenceSplitter'] is CppSentenceSplitter


def test_cpp_override_skips_missing_symbol(monkeypatch):
    cpp_module = types.ModuleType('lazyllm.lazyllm_cpp')
    monkeypatch.setitem(sys.modules, 'lazyllm.lazyllm_cpp', cpp_module)
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', 'true')

    marker = object()
    fake_globals = {'__name__': 'fake.module', 'SentenceSplitter': marker}
    applied = cpp.override_with_cpp_exports(fake_globals, ('SentenceSplitter',))

    assert applied == []
    assert fake_globals['SentenceSplitter'] is marker


def test_cpp_override_import_failure(monkeypatch):
    monkeypatch.setenv('LAZYLLM_ENABLE_CPP_OVERRIDE', 'on')
    monkeypatch.delitem(sys.modules, 'lazyllm.lazyllm_cpp', raising=False)

    real_import_module = cpp.importlib.import_module

    def _raise_import_error(name):
        if name == 'lazyllm.lazyllm_cpp':
            raise ImportError('mock import error')
        return real_import_module(name)

    monkeypatch.setattr(cpp.importlib, 'import_module', _raise_import_error)

    fake_globals = {'__name__': 'fake.module', 'SentenceSplitter': object()}
    assert cpp.override_with_cpp_exports(fake_globals, ('SentenceSplitter',)) == []
