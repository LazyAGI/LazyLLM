import os
import uuid

import pytest

from lazyllm import config
from lazyllm.common.queue import FileSystemQueue, RedisQueue, SQLiteQueue
from lazyllm.thirdparty import redis


@pytest.fixture(params=['sqlite', 'redis'])
def queue(request, tmp_path):
    url = os.environ.get('LAZYLLM_TEST_REDIS_URL', config['fsqredis_url'])
    if request.param == 'redis' and not url:
        pytest.skip('Set LAZYLLM_TEST_REDIS_URL to run the Redis queue contract tests')
    klass = 'queue-contract-' + uuid.uuid4().hex
    with config.temp('home', str(tmp_path)), config.temp('fsqredis_url', url):
        cls = RedisQueue if request.param == 'redis' else SQLiteQueue
        value = cls(klass=klass)
        try:
            yield value
        finally:
            if request.param == 'redis':
                redis.Redis.from_url(url).delete(value.sid)
            FileSystemQueue.__queue_pool__.pop(klass, None)


@pytest.mark.parametrize('limit', [1, 2, 10])
def test_limited_dequeue_preserves_fifo_and_queue_size(queue, limit):
    messages = ['A', 'B', 'C']
    for message in messages:
        queue.enqueue(message)
    count = min(limit, len(messages))
    assert queue.dequeue(limit=limit) == messages[:count]
    assert queue.size() == len(messages) - count
    assert queue.dequeue() == messages[count:]
    queue.enqueue('next')
    assert queue.peek() == 'next'
    assert queue.dequeue() == ['next']


@pytest.mark.parametrize('reset', ['clear', 'init'])
def test_queue_can_be_reused_after_reset(queue, reset):
    queue.enqueue('old')
    getattr(queue, reset)()
    assert queue.size() == 0
    assert queue.peek() is None
    assert queue.dequeue() == []
    queue.enqueue('new')
    assert queue.size() == 1
    assert queue.peek() == 'new'
    assert queue.dequeue() == ['new']


def test_empty_limited_dequeue_keeps_queue_reusable(queue):
    assert queue.dequeue(limit=1) == []
    assert queue.size() == 0
    queue.enqueue('first')
    assert queue.dequeue() == ['first']


def test_unlimited_dequeue_keeps_queue_reusable(queue):
    for message in ['你好', 'world']:
        queue.enqueue(message)
    assert queue.dequeue() == ['你好', 'world']
    assert queue.dequeue() == []
    queue.enqueue('again')
    assert queue.dequeue() == ['again']


def test_repeated_reset_keeps_queue_reusable(queue):
    queue.clear()
    queue.init()
    assert queue.size() == 0
    queue.enqueue('first')
    assert queue.dequeue(limit=10) == ['first']
    assert queue.size() == 0
