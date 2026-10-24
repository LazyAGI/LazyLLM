import asyncio
import uuid

import pytest

from lazyllm import config
from lazyllm.configs import Config


def test_temporary_value_restored_after_error():
    previous = config['repr_ml']
    with pytest.raises(RuntimeError, match='body'):
        with config.temp('repr_ml', not previous):
            assert config['repr_ml'] != previous
            raise RuntimeError('body')
    assert config['repr_ml'] == previous


def test_nested_temporary_values_restore_each_scope():
    previous = config['repr_ml']
    with config.temp('repr_ml', not previous):
        with pytest.raises(RuntimeError, match='inner'):
            with config.temp('repr_ml', previous):
                raise RuntimeError('inner')
        assert config['repr_ml'] != previous
    assert config['repr_ml'] == previous


@pytest.mark.parametrize('asynchronous', [False, True])
def test_namespace_restored_after_error(tmp_path, asynchronous):
    name = 'restore' + uuid.uuid4().hex
    target = Config(name, home=str(tmp_path))

    def body():
        previous = config._config
        with pytest.raises(RuntimeError, match='body'):
            with config.namespace(name):
                assert config._config is target
                raise RuntimeError('body')
        assert config._config is previous

    async def asynchronous_body():
        body()

    if asynchronous:
        asyncio.run(asynchronous_body())
    else:
        body()


@pytest.mark.parametrize('asynchronous', [False, True])
def test_normal_namespace_exit_control(tmp_path, asynchronous):
    name = 'normal' + uuid.uuid4().hex
    target = Config(name, home=str(tmp_path))

    def body():
        previous = config._config
        with config.namespace(name):
            assert config._config is target
        assert config._config is previous

    async def asynchronous_body():
        body()

    if asynchronous:
        asyncio.run(asynchronous_body())
    else:
        body()
