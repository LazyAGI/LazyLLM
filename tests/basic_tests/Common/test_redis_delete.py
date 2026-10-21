import os
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get('LAZYLLM_REDIS_DELETE_TEST_URL'),
    reason='Requires an explicitly supplied isolated Redis via LAZYLLM_REDIS_URL',
)


def test_prefixed_single_delete_leaves_other_namespace():
    from lazyllm.common.redis_client import RedisClient
    root = RedisClient(uuid.uuid4().hex)
    owner, other = root['owner'], root['other']
    owner.set('key', 'owned')
    other.set('key', 'other')
    try:
        owner.delete('key')
        assert owner.get('key') is None
        assert other.get('key') == b'other'
    finally:
        other.delete('key')


def test_multiple_keys_are_deleted_with_prefix():
    from lazyllm.common.redis_client import RedisClient
    client = RedisClient(uuid.uuid4().hex)
    client.set('first', 'one')
    client.set('second', 'two')
    client.delete(['first', 'second'])
    assert client.get('first') is None
    assert client.get('second') is None


def test_redis_globals_consumes_its_serialized_entry():
    from lazyllm.common.globals import RedisGlobals
    state = RedisGlobals()
    state['regression'] = 'payload'
    key = state.pickled_data()
    state.clear()
    state.unpickle_and_update_data(key)
    assert state['regression'] == 'payload'
    assert state._redis_client.get(state._get_redis_key(key)) is None
