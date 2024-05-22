import sys
import pytest

class TestFn_Thirdparty(object):
    
    def test_import(self, monkeypatch):
        def check_installed():
            try:
                import llama_index
                return True
            except ImportError:
                return False
        monkeypatch.delitem(sys.modules, "llama_index", raising=False)
        assert check_installed() == False

    def test_lazy_import(self, monkeypatch):
        def check_lazy_import(llama_index):
            try:
                llama_index.a
                return True
            except ImportError:
                return False
        monkeypatch.delitem(sys.modules, "llama_index", raising=False)
        from lazyllm.thirdparty import llama_index
        assert llama_index is not None
        assert check_lazy_import(llama_index) == False
