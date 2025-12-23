import sys
import pytest
from lazyllm.thirdparty import faiss, requirements, prep_req_dict

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
            except (AttributeError, ImportError):
                return False
        monkeypatch.delitem(sys.modules, "faiss", raising=False)
        assert faiss is not None
        assert not check_lazy_import(faiss)

    def test_lazy_import_with_path(self):
        class Flag(object): pass
        flag = Flag()
        flag.flag = False

        def patch():
            flag.flag = True

        from lazyllm.thirdparty import mineru
        mineru.register_patches(patch)
        assert not flag.flag
        cli = mineru.cli
        common = cli.common
        assert not flag.flag

        with pytest.raises(ImportError):
            _ = common.aio_do_parse
        assert not flag.flag

        from lazyllm.thirdparty import os
        os.register_patches(patch)
        path = os.path
        assert not flag.flag
        _ = path.join
        assert flag.flag

    def test_toml_dependencies_extraction(self):
        prep_req_dict()
        assert requirements
        for name, version in requirements.items():
            print(f'{name} {version}')
