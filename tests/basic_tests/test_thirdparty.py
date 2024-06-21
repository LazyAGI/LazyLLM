import sys
from lazyllm.thirdparty import llama_index
import pytest

class TestFn_Thirdparty(object):
    
    def test_import(self, monkeypatch):
        def check_installed(third_import_type):
            try:
                import llama_index
                # if env install real llama_index
                return third_import_type == type(llama_index)
            except ImportError:
                return False
        third_import_type = type(llama_index)
        monkeypatch.delitem(sys.modules, "llama_index", raising=False)
        assert check_installed(third_import_type) == False

    def test_lazy_import(self, monkeypatch):
        def check_lazy_import(llama_index):
            try:
                llama_index.a
                return True
            except AttributeError:
                return False
        monkeypatch.delitem(sys.modules, "llama_index", raising=False)
        assert llama_index is not None
        assert check_lazy_import(llama_index) == False
