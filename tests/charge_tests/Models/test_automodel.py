import os

import pytest

import lazyllm
from lazyllm import AutoModel
from lazyllm.module.llms import automodel as automodel_module
from lazyllm.module.llms.utils import get_module_config_map

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'automodel_config.yaml')


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv('LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH', raising=False)
    monkeypatch.delenv('LAZYLLM_SENSENOVA_API_KEY', raising=False)
    get_module_config_map.cache_clear()
    monkeypatch.setitem(lazyllm.config.impl, 'auto_model_config_map_path', CONFIG_PATH)
    lazyllm.config.refresh('sensenova_api_key')


@pytest.fixture
def dummy_modules(monkeypatch):
    class DummyOnline:
        instances = []

        def __init__(self, **kwargs):
            recorded = dict(kwargs)
            if recorded.get('source') == 'sensenova' and 'api_key' not in recorded:
                api_key = lazyllm.config['sensenova_api_key']
                if api_key:
                    recorded['api_key'] = api_key
            self.kwargs = recorded
            DummyOnline.instances.append(self)

    class DummyTrainable:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            DummyTrainable.instances.append(self)

    monkeypatch.setattr(automodel_module, 'OnlineModule', DummyOnline)
    monkeypatch.setattr(automodel_module, 'TrainableModule', DummyTrainable)
    DummyOnline.instances.clear()
    DummyTrainable.instances.clear()
    return DummyOnline, DummyTrainable


class TestAutoModel(object):

    def test_autmodel_defaults_to_trainable(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules

        result = AutoModel(model='internlm-test')

        assert isinstance(result, DummyTrainable)
        assert len(DummyOnline.instances) == 0
        assert DummyTrainable.instances[0].kwargs['base_model'] == 'internlm-test'
        assert os.environ['LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH'] == CONFIG_PATH

    def test_autmodel_prefers_online_when_env_key_available(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules
        monkeypatch.setenv('LAZYLLM_SENSENOVA_API_KEY', 'env-key')
        lazyllm.config.refresh('LAZYLLM_SENSENOVA_API_KEY')

        result = AutoModel(model='sensenova-model')

        assert isinstance(result, DummyOnline)
        assert len(DummyTrainable.instances) == 0
        kwargs = DummyOnline.instances[0].kwargs
        assert kwargs['source'] == 'sensenova'
        assert kwargs['url'] == 'https://api.sensenova.com/v1/'
        assert kwargs['api_key'] == 'env-key'

    def test_autmodel_respects_explicit_source(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules

        result = AutoModel(model='glm-model', source='glm')

        assert isinstance(result, DummyOnline)
        assert len(DummyTrainable.instances) == 0
        kwargs = DummyOnline.instances[0].kwargs
        assert kwargs['source'] == 'glm'
        assert kwargs['url'] == 'https://glm.fake.endpoint/v1/'

    def test_autmodel_uses_trainable_when_config_entry_has_framework(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules

        result = AutoModel(model='trainable-model')

        assert isinstance(result, DummyTrainable)
        assert len(DummyOnline.instances) == 0
        assert os.environ['LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH'] == CONFIG_PATH

    def test_autmodel_uses_online_entry_url_and_port(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules

        result = AutoModel(model='online-url-model')

        assert isinstance(result, DummyOnline)
        assert len(DummyTrainable.instances) == 0
        kwargs = DummyOnline.instances[0].kwargs
        assert kwargs['url'] == 'http://custom.online.endpoint/v1/'
        assert kwargs['port'] == 9001

    def test_autmodel_uses_configured_online_credentials(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules

        result = AutoModel(model='credential-model')

        assert isinstance(result, DummyOnline)
        kwargs = DummyOnline.instances[0].kwargs
        assert kwargs['source'] == 'sensenova'
        assert kwargs['api_key'] == 'config-key'

    def test_autmodel_reads_env_key_when_config_lacks_api_key(self, monkeypatch, dummy_modules):
        DummyOnline, DummyTrainable = dummy_modules
        monkeypatch.setenv('LAZYLLM_SENSENOVA_API_KEY', 'env-key')
        lazyllm.config.refresh('LAZYLLM_SENSENOVA_API_KEY')

        result = AutoModel(model='env-only-model')

        assert isinstance(result, DummyOnline)
        kwargs = DummyOnline.instances[0].kwargs
        assert kwargs['source'] == 'sensenova'
        assert kwargs['api_key'] == 'env-key'
