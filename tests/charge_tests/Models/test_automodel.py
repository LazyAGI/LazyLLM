import os
from types import SimpleNamespace

import pytest

import lazyllm
from lazyllm import AutoModel
from lazyllm.module.llms import automodel as automodel_module
from lazyllm.module.llms import online_module as online_module

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'automodel_config.yaml')


class CallRecorder:
    def __init__(self):
        self.calls = []

    def record(self, name, kwargs):
        self.calls.append((name, kwargs))

    def last(self, name):
        for kind, kwargs in reversed(self.calls):
            if kind == name:
                return kwargs
        return None


def set_sensenova_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('LAZYLLM_SENSENOVA_API_KEY', raising=False)
    else:
        monkeypatch.setenv('LAZYLLM_SENSENOVA_API_KEY', value)
    lazyllm.config.refresh('sensenova_api_key')


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv('LAZYLLM_TRAINABLE_MODULE_CONFIG_MAP_PATH', raising=False)
    monkeypatch.delenv('LAZYLLM_SENSENOVA_API_KEY', raising=False)
    monkeypatch.setitem(lazyllm.config.impl, 'auto_model_config_map_path', CONFIG_PATH)
    lazyllm.config.refresh('sensenova_api_key')


@pytest.fixture
def fake_modules(monkeypatch):
    recorder = CallRecorder()

    class FakeOnlineModule:
        def __init__(self, **kwargs):
            recorded = dict(kwargs)
            source = recorded.get('source')
            env_key = os.environ.get('LAZYLLM_SENSENOVA_API_KEY')
            if not source and not recorded.get('url') and not recorded.get('base_url') and not env_key:
                raise KeyError('No api_key is configured for any of the models.')
            if not source and env_key:
                recorded['source'] = 'sensenova'
                source = 'sensenova'
            if source == 'sensenova' and 'api_key' not in recorded and env_key:
                recorded['api_key'] = env_key
            recorder.record('online', recorded)
            self.kwargs = recorded

    class FakeTrainableModule:
        def __init__(self, base_model=None, **kwargs):
            recorded = dict(kwargs)
            if base_model is not None:
                recorded.setdefault('base_model', base_model)
            recorder.record('trainable', recorded)
            self.kwargs = recorded
            self._url = recorded.get('target_path') or recorded.get('url') or 'local'
            self._impl = SimpleNamespace(_get_deploy_tasks=SimpleNamespace(flag=True))

    class FakeOnlineChatModule:
        def __init__(self, **kwargs):
            recorder.record('online_chat', dict(kwargs))
            self.kwargs = dict(kwargs)

    monkeypatch.setattr(automodel_module, 'OnlineModule', FakeOnlineModule)
    monkeypatch.setattr(automodel_module, 'TrainableModule', FakeTrainableModule)
    monkeypatch.setattr(online_module, 'OnlineChatModule', FakeOnlineChatModule)
    return SimpleNamespace(
        recorder=recorder,
        FakeOnlineModule=FakeOnlineModule,
        FakeTrainableModule=FakeTrainableModule,
        FakeOnlineChatModule=FakeOnlineChatModule,
    )


# Case 1: no config/env/source -> TrainableModule
def test_automodel_defaults_to_trainable_without_config_env_source(monkeypatch, fake_modules):
    set_sensenova_env(monkeypatch, None)

    result = AutoModel(model='no-config-model', config=False)

    recorder = fake_modules.recorder
    assert isinstance(result, fake_modules.FakeTrainableModule)
    assert recorder.last('online') is None
    assert recorder.last('trainable')['base_model'] == 'no-config-model'


# Case 2: env key only -> sensenova OnlineModule
def test_automodel_env_key_routes_to_sensenova_online(monkeypatch, fake_modules):
    set_sensenova_env(monkeypatch, 'env-key')

    result = AutoModel(model='no-config-model', config=False)

    recorder = fake_modules.recorder
    assert isinstance(result, fake_modules.FakeOnlineModule)
    assert recorder.last('trainable') is None
    kwargs = recorder.last('online')
    assert kwargs['source'] == 'sensenova'
    assert kwargs['api_key'] == 'env-key'


# Case 3: explicit source -> OnlineModule
def test_automodel_respects_explicit_source(monkeypatch, fake_modules):
    set_sensenova_env(monkeypatch, None)

    result = AutoModel(model='explicit-source-model', source='glm', config=False)

    recorder = fake_modules.recorder
    assert isinstance(result, fake_modules.FakeOnlineModule)
    assert recorder.last('trainable') is None
    kwargs = recorder.last('online')
    assert kwargs['source'] == 'glm'


@pytest.mark.parametrize(
    'model, env_key, expected_kind, expected_fields',
    [
        (
            'trainable-model',
            None,
            'trainable',
            {
                'base_model': 'trainable-model',
                'source': 'local',
                'target_path': 'http://127.0.0.1:2333/v1/',
                'use_model_map': CONFIG_PATH,
            },
        ),
        (
            'online-url-model',
            None,
            'online',
            {
                'model': 'online-url-model',
                'url': 'http://custom.online.endpoint/v1/',
            },
        ),
        (
            'credential-model',
            None,
            'online',
            {
                'source': 'sensenova',
                'api_key': 'config-key',
                'url': 'https://credential.endpoint/v1/',
            },
        ),
        (
            'env-only-model',
            'env-key',
            'online',
            {
                'source': 'sensenova',
                'api_key': 'env-key',
            },
        ),
    ],
    ids=[
        'case4_trainable_framework_url',
        'case5_online_url_only',
        'case6_online_sensenova_api_key',
        'case7_online_sensenova_env_key',
    ],
)
def test_automodel_config_routing(model, env_key, expected_kind, expected_fields, monkeypatch, fake_modules):
    set_sensenova_env(monkeypatch, env_key)

    result = AutoModel(model=model, config=CONFIG_PATH)

    recorder = fake_modules.recorder
    if expected_kind == 'trainable':
        assert isinstance(result, fake_modules.FakeTrainableModule)
        assert recorder.last('online') is None
        kwargs = recorder.last('trainable')
    else:
        assert isinstance(result, fake_modules.FakeOnlineModule)
        assert recorder.last('trainable') is None
        kwargs = recorder.last('online')

    for key, value in expected_fields.items():
        assert kwargs[key] == value
