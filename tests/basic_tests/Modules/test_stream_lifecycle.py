import asyncio
import json
import threading
import uuid
from contextvars import ContextVar

import pytest

import lazyllm
from lazyllm.module import stream_helper
from lazyllm.tools.writer.tools import stream_tools


@pytest.fixture(autouse=True)
def session():
    sid = f'stream-test-{uuid.uuid4().hex}'
    with lazyllm.globals._bind_sid(sid), lazyllm.locals._scope():
        yield
    lazyllm.globals._clear_sid(sid)


def emit():
    lazyllm.FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': 'ready'}))


@pytest.mark.parametrize('asynchronous', [False, True])
def test_stream_normal_completion_inherits_state_and_does_not_cancel(asynchronous):
    marker = ContextVar('stream-context')
    marker.set('permission')
    state = lazyllm.locals['_lazyllm_agent']
    state['workspace'] = {'history': ['parent']}
    called = []

    def work():
        assert marker.get() == 'permission'
        assert lazyllm.locals['_lazyllm_agent'] is state
        emit()
        return 'result'

    helper = lazyllm.StreamCallHelper(work, init_sid=False, on_cancel=lambda: called.append(True))
    helper.close()  # Closing an unstarted helper does not prevent subsequent use.

    async def consume():
        values = [item async for item in helper.astream()]
        await helper.aclose()
        return values

    values = asyncio.run(consume()) if asynchronous else list(helper())
    helper.close()
    assert values == [{'tag': 'text', 'delta': 'ready'}]
    assert helper.future.result() == 'result'
    assert called == []


@pytest.mark.parametrize('callback_fails', [False, True])
def test_sync_early_close_waits_and_notifies_once(callback_fails):
    release, cancelled, finished = threading.Event(), threading.Event(), threading.Event()
    notifications = []

    def work():
        emit()
        assert release.wait(5)
        finished.set()
        raise asyncio.CancelledError('producer')

    def cancel():
        notifications.append(True)
        cancelled.set()
        if callback_fails:
            raise ValueError('cancel callback failed')

    helper = lazyllm.StreamCallHelper(work, init_sid=False, on_cancel=cancel)
    iterator = helper()
    assert next(iterator)['delta'] == 'ready'
    closer = threading.Thread(target=iterator.close)
    closer.start()
    try:
        assert cancelled.wait(2)
        assert closer.is_alive()
        assert not finished.is_set()
    finally:
        release.set()
        closer.join(5)
    assert not closer.is_alive()
    assert finished.is_set()
    helper.close()
    assert notifications == [True]
    with pytest.raises(asyncio.CancelledError, match='producer'):
        helper.future.result()


def test_async_repeated_cancel_waits_for_producer_and_preserves_first_reason():
    async def scenario():
        release = threading.Event()
        notified = asyncio.Event()
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()
        calls = []

        def work():
            emit()
            assert release.wait(5)
            return 'stopped'

        def cancel():
            calls.append(True)
            loop.call_soon_threadsafe(notified.set)
            raise RuntimeError('callback failure must not mask cancellation')

        helper = lazyllm.StreamCallHelper(work, interval=0.001, init_sid=False, on_cancel=cancel)

        async def consume():
            async for _ in helper.astream():
                ready.set()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(ready.wait(), 2)
            task.cancel('first cancellation')
            await asyncio.wait_for(notified.wait(), 2)
            task.cancel('second cancellation')
            await asyncio.sleep(0)
            assert not task.done()
            assert not helper.future.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError, match='first cancellation'):
            await task
        assert helper.future.done()
        assert calls == [True]
        await helper.aclose()

    asyncio.run(scenario())


def test_queued_stream_cancel_never_calls_producer_or_callback(monkeypatch):
    gate, entered = threading.Event(), threading.Event()
    calls = []
    with lazyllm.ThreadPoolExecutor(max_workers=1, local_scope='inherit') as pool:
        monkeypatch.setattr(stream_helper, '_g_stream_thread_pool', pool)

        def block():
            entered.set()
            assert gate.wait(5)

        pool.submit(block)
        assert entered.wait(2)
        helper = lazyllm.StreamCallHelper(
            lambda: calls.append('work'), init_sid=False, on_cancel=lambda: calls.append('cancel'))
        try:
            helper._submit()
            helper.close()
            assert helper.future.cancelled()
            assert calls == []
        finally:
            gate.set()


@pytest.mark.parametrize('reason', ['close', 'error', 'explicit_cancel'])
def test_async_close_waits_and_preserves_exit_reason(reason):
    async def scenario():
        release = threading.Event()
        notified = asyncio.Event()
        loop = asyncio.get_running_loop()

        def work():
            emit()
            assert release.wait(5)
            raise ValueError('producer failure stays in future')

        helper = lazyllm.StreamCallHelper(
            work, init_sid=False, on_cancel=lambda: loop.call_soon_threadsafe(notified.set))
        iterator = helper.astream()
        assert (await anext(iterator))['delta'] == 'ready'
        original = ValueError('consumer failed')
        operation = (iterator.athrow(original) if reason == 'error' else
                     helper.aclose() if reason == 'explicit_cancel' else iterator.aclose())
        task = asyncio.create_task(operation)
        try:
            await asyncio.wait_for(notified.wait(), 2)
            assert not task.done()
            task.cancel('cancel during close')
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        if reason == 'error':
            with pytest.raises(ValueError) as raised:
                await task
            assert raised.value is original
        elif reason == 'explicit_cancel':
            with pytest.raises(asyncio.CancelledError, match='cancel during close'):
                await task
        else:
            await task
        await iterator.aclose()
        with pytest.raises(ValueError, match='producer failure'):
            helper.future.result()

    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['close', 'consume', 'idle'])
def test_writer_waits_and_raises_cancellation_in_producer(failure):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    consumer_done = threading.Event()
    errors = []
    parent = lazyllm.locals['_lazyllm_agent']

    def work(sink):
        assert lazyllm.locals['_lazyllm_agent'] is parent
        entered.set()
        try:
            assert release.wait(5)
            sink({'delta': 'late'})
        finally:
            finished.set()

    def consume(_payload):
        raise ValueError('consume failed')

    stream = stream_tools.DraftPreviewStream(
        work, consume, lambda result: ([], {}), idle_timeout=0.02,
        initial_deltas=['ready'],
    )
    try:
        assert entered.wait(2)
        assert next(stream) == 'ready'

        def close_or_consume():
            try:
                if failure == 'close':
                    stream.close()
                else:
                    if failure == 'consume':
                        stream._queue.put({'delta': 'bad'})
                    list(stream)
            except BaseException as error:
                errors.append(error)
            finally:
                consumer_done.set()

        consumer = threading.Thread(target=close_or_consume)
        consumer.start()
        # The event also synchronizes the producer-side cancellation checkpoint.
        assert stream._cancelled.wait(2)
        assert not consumer_done.is_set()
    finally:
        release.set()
        if 'consumer' in locals():
            consumer.join(5)
        stream.close()
    assert finished.is_set()
    assert consumer_done.is_set()
    assert len(errors) == (failure != 'close')
    if errors:
        assert isinstance(errors[0], ValueError if failure == 'consume' else TimeoutError)
    with pytest.raises(asyncio.CancelledError):
        stream._future.result()
