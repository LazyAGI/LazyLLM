import sys
import pytest
from lazyllm.thirdparty import faiss, requirements
from lazyllm import thirdparty

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
        monkeypatch.delitem(sys.modules, 'faiss', raising=False)
        assert not check_installed(third_import_type)

    def test_lazy_import(self, monkeypatch):
        def check_lazy_import(faiss):
            try:
                faiss.a
                return True
            except (AttributeError, ImportError):
                return False
        monkeypatch.delitem(sys.modules, 'faiss', raising=False)
        assert faiss is not None
        assert not check_lazy_import(faiss)

    def test_lazy_import_with_path(self):
        class Flag(object): pass
        flag = Flag()
        flag.flag = False

        def patch():
            flag.flag = True

        from lazyllm.thirdparty import graphrag
        graphrag.register_patches(patch)
        assert not flag.flag
        load_config = graphrag.config.load_config
        assert not flag.flag

        with pytest.raises(ImportError):
            _ = load_config.load_config
        assert not flag.flag

        from lazyllm.thirdparty import os
        os.register_patches(patch)
        path = os.path
        assert not flag.flag
        _ = path.join
        assert flag.flag

    def test_toml_dependencies_extraction(self):
        thirdparty.prepare_requirements_dict()
        assert requirements

    def test_check_package_installed(self):
        assert thirdparty.check_package_installed('lazyllm')
        assert thirdparty.check_package_installed(['lazyllm', 'requests'])
        assert not thirdparty.check_package_installed(['lazyllm', 'requests', 'nonexistent_module_kasduf45123'])
        assert not thirdparty.check_package_installed('nonexistent_module_kasduf45123')

    def test_load_toml_dep_group(self):
        assert len(thirdparty.load_toml_dep_group('full')) > 0

    def test_check_dependency_by_group(self):
        try:
            assert thirdparty.check_dependency_by_group('standard')
        except ImportError:
            assert True, 'Normal exit due to missing dependencies'

    def test_python_multipart_resolves_to_its_import_name(self):
        # its import name (multipart) differs from its PyPI name
        assert thirdparty.package_name_map.get('multipart') == 'python-multipart'
        assert thirdparty.package_name_map_reverse.get('python-multipart') == 'multipart'

    def test_import_error_hint_when_module_not_installed(self):
        w = thirdparty.PackageWrapper('nonexistent_module_kasduf45123')
        with pytest.raises(ImportError) as exc_info:
            _ = w.prop
        msg = str(exc_info.value)
        assert 'Cannot import module `nonexistent_module_kasduf45123`' in msg
        assert 'please install it by `pip install nonexistent_module_kasduf45123`' in msg

    def test_import_error_hint_when_installed_but_internal_failure(self, monkeypatch):
        def broken_import(*args, **kwargs):
            raise ImportError('failed to resolve old huggingface-hub dependency')
        monkeypatch.setattr('importlib.import_module', broken_import)
        w = thirdparty.PackageWrapper('transformers')
        with pytest.raises(ImportError) as exc_info:
            _ = w.model
        msg = str(exc_info.value)
        assert 'Module `transformers` is installed, but importing it failed' in msg
        assert 'huggingface-hub' in msg
        assert 'old huggingface-hub dependency' in msg
