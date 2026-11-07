import os
import uuid

import pytest

from lazyllm.common.queue import FileSystemQueue, RedisQueue


@pytest.fixture
def queue():
    if not os.environ.get('LAZYLLM_FSQREDIS_URL'):
        pytest.skip('requires a dedicated Redis server via LAZYLLM_FSQREDIS_URL')
    name = 'dequeue-limit-' + uuid.uuid4().hex
    queue = RedisQueue(klass=name)
    try:
        yield queue
    finally:
        queue.clear()
        FileSystemQueue.__queue_pool__.pop(name)


@pytest.mark.parametrize('limit', [1, 2, 5])
def test_limit_preserves_fifo_size_and_reuse(queue, limit):
    values = ['first', 'second', 'third']
    for value in values:
        queue.enqueue(value)
    expected = values[:limit]
    assert queue.dequeue(limit=limit) == expected
    remaining = values[len(expected):]
    assert queue.size() == len(remaining)
    assert queue.peek() == (remaining[0] if remaining else None)
    assert queue.dequeue() == remaining
    assert queue.size() == 0
    assert queue.dequeue(limit=limit) == []
    queue.enqueue('after-empty')
    assert queue.size() == 1
    assert queue.dequeue(limit=limit) == ['after-empty']
    assert queue.size() == 0


def test_repeated_single_item_dequeue_has_no_duplicates(queue):
    for value in ['a', 'b', 'c']:
        queue.enqueue(value)
    assert [queue.dequeue(limit=1) for _ in range(4)] == [['a'], ['b'], ['c'], []]
    assert queue.size() == 0


@pytest.mark.parametrize('limit', [None, 0])
def test_unlimited_dequeue_keeps_existing_behavior(queue, limit):
    queue.enqueue('a')
    queue.enqueue('b')
    assert queue.dequeue(limit=limit) == ['a', 'b']
    assert queue.size() == 0
    queue.enqueue('c')
    assert queue.dequeue() == ['c']
