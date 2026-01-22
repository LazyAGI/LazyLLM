import os

import pytest

import lazyllm

from tests.charge_tests import conftest as charge_conftest


class _Marker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Node:
    def __init__(self, markers):
        self._markers = markers

    def get_closest_marker(self, name):
        return self._markers.get(name)


class _Config:
    def __init__(self, changed_files):
        self.changed_files = changed_files


class _Request:
    def __init__(self, node, config):
        self.node = node
        self.config = config


def _run_cache_policy_fixture(request):
    gen = charge_conftest.ignore_cache_on_change(request)
    next(gen)
    try:
        yield
    finally:
        with pytest.raises(StopIteration):
            next(gen)


def test_advanced_test_always_uses_cache(monkeypatch):
    old_env = os.environ.get('LAZYLLM_CACHE_ONLINE_MODULE')
    old_value = lazyllm.config['cache_online_module']

    node = _Node(
        markers={
            'advanced_test': _Marker(),
            'ignore_cache_on_change': _Marker('a.py'),
        }
    )
    request = _Request(node=node, config=_Config(changed_files=['a.py']))

    with _run_cache_policy_fixture(request):
        assert lazyllm.config['cache_online_module'] is True

    assert lazyllm.config['cache_online_module'] == old_value
    assert os.environ.get('LAZYLLM_CACHE_ONLINE_MODULE') == old_env


def test_connectivity_linux_disables_cache_when_changed(monkeypatch):
    monkeypatch.setattr(charge_conftest.sys, 'platform', 'linux')

    node = _Node(
        markers={
            'model_connectivity_test': _Marker(),
            'ignore_cache_on_change': _Marker('a.py'),
        }
    )
    request = _Request(node=node, config=_Config(changed_files=['a.py']))

    with _run_cache_policy_fixture(request):
        assert lazyllm.config['cache_online_module'] is False


def test_connectivity_linux_uses_cache_when_not_changed(monkeypatch):
    monkeypatch.setattr(charge_conftest.sys, 'platform', 'linux')

    node = _Node(markers={'model_connectivity_test': _Marker()})
    request = _Request(node=node, config=_Config(changed_files=['a.py']))

    with _run_cache_policy_fixture(request):
        assert lazyllm.config['cache_online_module'] is True


def test_connectivity_non_linux_always_uses_cache(monkeypatch):
    monkeypatch.setattr(charge_conftest.sys, 'platform', 'darwin')

    node = _Node(
        markers={
            'model_connectivity_test': _Marker(),
            'ignore_cache_on_change': _Marker('a.py'),
        }
    )
    request = _Request(node=node, config=_Config(changed_files=['a.py']))

    with _run_cache_policy_fixture(request):
        assert lazyllm.config['cache_online_module'] is True
