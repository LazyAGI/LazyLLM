import sys
from lazyllm.thirdparty import faiss

class TestThirdparty(object):

    def test_import(self, monkeypatch):
        def check_installed(third_import_type):
            try:
                import faiss
                # if env install real llama_index
                return third_import_type == type(faiss)
            except ImportError:
                return False
        third_import_type = type(faiss)
        monkeypatch.delitem(sys.modules, "faiss", raising=False)
        assert not check_installed(third_import_type)

    def test_lazy_import(self, monkeypatch):
        def check_lazy_import(faiss):
            try:
                faiss.a
                return True
            except AttributeError:
                return False
        monkeypatch.delitem(sys.modules, "faiss", raising=False)
        assert faiss is not None
        assert not check_lazy_import(faiss)
