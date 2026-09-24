import asyncio
from contextvars import ContextVar
import threading
import uuid

import pytest

import lazyllm
from lazyllm.flow.flow import Parallel


@pytest.fixture(autouse=True)
def request_scope():
    old_global, old_local = lazyllm.globals._sid, lazyllm.locals._sid
    store = lazyllm.locals._MemoryGlobals__data
    before = set(store)
    lazyllm.globals._init_sid(f'test-global-{uuid.uuid4().hex}')
    lazyllm.locals._init_sid(f'test-local-{uuid.uuid4().hex}')
    yield
    lazyllm.globals.clear()
    for sid in set(store) - before:
        store.pop(sid, None)
    lazyllm.globals._init_sid(old_global)
    lazyllm.locals._init_sid(old_local)


def test_reused_worker_isolates_each_submission_without_agent_cleanup():
    parent = lazyllm.locals['_lazyllm_agent']
    parent['workspace'] = {'marker': 'parent'}

    def run(marker):
        state = lazyllm.locals['_lazyllm_agent']
        previous = state.get('workspace')
        state['workspace'] = {'marker': marker}
        return threading.get_ident(), lazyllm.globals._sid, lazyllm.locals._sid, previous

    with lazyllm.ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(run, 'A').result(timeout=5)
        lazyllm.globals._init_sid('test-global-B')
        second = pool.submit(run, 'B').result(timeout=5)
    assert first[0] == second[0]
    assert second[1] == 'test-global-B'
    assert first[2] != second[2]
    assert first[3] is second[3] is None
    assert parent['workspace'] == {'marker': 'parent'}


def test_inherit_uses_actual_local_sid_and_preserves_retained_state():
    local_a = lazyllm.locals._sid
    agent_a = lazyllm.locals['_lazyllm_agent']
    agent_a['workspace'] = {'marker': 'A'}
    with lazyllm.ThreadPoolExecutor(max_workers=1, local_scope='inherit') as pool:
        def read():
            return lazyllm.globals._sid, lazyllm.locals._sid, lazyllm.locals['_lazyllm_agent']

        first = pool.submit(read).result(timeout=5)
        lazyllm.locals._init_sid('test-local-B')
        lazyllm.locals['_lazyllm_agent']['workspace'] = {'marker': 'B'}
        second = pool.submit(read).result(timeout=5)
        lazyllm.locals._init_sid(local_a)
        third = pool.submit(read).result(timeout=5)
    assert first[0] != first[1]
    assert first[1] == third[1] == local_a
    assert first[2] is third[2] is agent_a
    assert second[2]['workspace'] == {'marker': 'B'}
    assert third[2]['workspace'] == {'marker': 'A'}


def test_concurrent_isolated_siblings_copy_context_but_not_local_workspace():
    marker = ContextVar('request-marker')
    marker.set('parent')
    parent_sid = lazyllm.locals._sid
    lazyllm.locals['_lazyllm_agent']['workspace'] = {'parent': True}
    barrier = threading.Barrier(2)

    def run(value):
        original = marker.get(None)
        marker.set(value)
        agent = lazyllm.locals['_lazyllm_agent']
        agent['workspace'] = {'value': value}
        barrier.wait(timeout=5)
        return original, marker.get(), lazyllm.locals._sid, agent['workspace']

    with lazyllm.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, value) for value in ('A', 'B')]
        a, b = [future.result(timeout=5) for future in futures]
    assert a[:2] == ('parent', 'A')
    assert b[:2] == ('parent', 'B')
    assert len({parent_sid, a[2], b[2]}) == 3
    assert a[3] is not b[3]
    assert marker.get() == 'parent'


def test_nested_submit_map_initializer_and_target_keyword_are_preserved():
    initialized = threading.local()
    marker = ContextVar('nested-marker', default=None)
    marker.set('root')

    def initialize(value):
        initialized.value = value

    def child(value, *, local_scope='target-argument'):
        return value, initialized.value, marker.get(), lazyllm.locals._sid, local_scope

    with lazyllm.ThreadPoolExecutor(1, initializer=initialize, initargs=('ready',)) as child_pool:
        with lazyllm.ThreadPoolExecutor(1) as parent_pool:
            def parent():
                parent_sid = lazyllm.locals._sid
                result = child_pool.submit(child, 1, local_scope='passed-through').result(timeout=5)
                return parent_sid, result

            parent_sid, result = parent_pool.submit(parent).result(timeout=5)
        mapped = list(child_pool.map(child, [2, 3]))
    assert result[:3] == (1, 'ready', 'root')
    assert result[3] != parent_sid
    assert result[4] == 'passed-through'
    assert [row[0] for row in mapped] == [2, 3]
    assert mapped[0][3] != mapped[1][3]


@pytest.mark.parametrize('error', [None, RuntimeError('failed'), asyncio.CancelledError('cancelled')])
def test_owned_scope_reclaimed_even_if_target_changes_binding(error):
    store = lazyllm.locals._MemoryGlobals__data
    lazyllm.locals['_lazyllm_agent']
    before = set(store)
    owned = []

    def run():
        owned.append(lazyllm.locals._sid)
        lazyllm.locals['_lazyllm_agent']['workspace'] = {'marker': 'owned'}
        lazyllm.locals._init_sid('test-business-scope')
        lazyllm.locals['_lazyllm_agent']['workspace'] = {'marker': 'business'}
        if error is not None:
            raise error

    with lazyllm.ThreadPoolExecutor(1) as pool:
        future = pool.submit(run)
        if error is None:
            future.result(timeout=5)
        else:
            with pytest.raises(type(error)) as caught:
                future.result(timeout=5)
            assert caught.value is error
    assert owned[0] not in store
    assert set(store) - before == {'test-business-scope'}
    assert store['test-business-scope']['_lazyllm_agent']['workspace'] == {'marker': 'business'}


def test_queued_cancellation_does_not_allocate_a_local_scope():
    store = lazyllm.locals._MemoryGlobals__data
    lazyllm.locals['_lazyllm_agent']
    before = set(store)
    started, release = threading.Event(), threading.Event()

    def run():
        lazyllm.locals['_lazyllm_agent']
        started.set()
        assert release.wait(timeout=5)

    with lazyllm.ThreadPoolExecutor(1) as pool:
        running = pool.submit(run)
        try:
            assert started.wait(timeout=5)
            during = set(store)
            queued = pool.submit(lambda: pytest.fail('cancelled task ran'))
            assert queued.cancel()
            assert set(store) == during
        finally:
            release.set()
        running.result(timeout=5)
    assert set(store) == before


@pytest.mark.parametrize('factory', [lazyllm.Thread, lazyllm.ThreadPoolExecutor])
def test_invalid_scope_rejected_at_construction(factory):
    with pytest.raises(ValueError, match='local_scope'):
        factory(local_scope='invalid')


@pytest.mark.parametrize('hooks_type', [list, tuple])
def test_thread_captures_context_at_construction_and_preserves_hooks(hooks_type):
    marker = ContextVar('thread-marker')
    marker.set('constructed')
    local_sid = lazyllm.locals._sid
    seen = []
    hooks = hooks_type([lambda: seen.append(marker.get()), lambda: seen.append(lazyllm.locals._sid)])
    original = tuple(hooks)
    thread = lazyllm.Thread(target=lambda: marker.get(), prehook=hooks, local_scope='inherit')
    marker.set('started')
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not thread.q.empty()
    assert thread.get_result() == 'constructed'
    assert seen == ['constructed', local_sid]
    assert tuple(hooks) == original


@pytest.mark.parametrize('prehook', [False, True])
@pytest.mark.parametrize('error', [RuntimeError('failure'), asyncio.CancelledError('cancel')])
def test_thread_transports_target_and_prehook_failures(prehook, error):
    def fail():
        raise error

    thread = lazyllm.Thread(target=(lambda: None) if prehook else fail, prehook=fail if prehook else None)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not thread.q.empty(), 'get_result would block after the worker exited'
    with pytest.raises(type(error)) as caught:
        thread.get_result()
    assert caught.value is error


def test_thread_can_return_an_exception_object():
    result = RuntimeError('a value')
    thread = lazyllm.Thread(target=lambda: result)
    thread.start()
    thread.join(timeout=5)
    assert thread.get_result() is result


@pytest.mark.parametrize('initialization_failure', [False, True])
def test_parallel_reclaims_scope_on_cancellation_or_copy_failure(initialization_failure):
    store = lazyllm.locals._MemoryGlobals__data
    parent_sid = lazyllm.locals._sid
    local_data = lazyllm.locals._data
    before = set(store)

    class BadCopy:
        def copy(self):
            raise ValueError('copy failed')

    def fail():
        raise asyncio.CancelledError('cancelled')

    with pytest.raises(ValueError if initialization_failure else asyncio.CancelledError):
        Parallel._worker(fail, None, lazyllm.globals._sid,
                         {'bad': BadCopy()} if initialization_failure else local_data)
    assert set(store) == before
    assert lazyllm.locals._sid == parent_sid


def test_parallel_retains_shallow_workspace_feedback():
    workspace = {'activated': []}
    lazyllm.locals['_lazyllm_agent']['workspace'] = workspace
    parent_sid = lazyllm.locals._sid

    def activate(value):
        assert lazyllm.locals._sid != parent_sid
        lazyllm.locals['_lazyllm_agent']['workspace']['activated'].append(value)
        return value

    assert tuple(lazyllm.diverter(activate, activate)(('A', 'B'))) == ('A', 'B')
    assert sorted(workspace['activated']) == ['A', 'B']
