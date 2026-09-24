import asyncio

import pytest

import lazyllm


def assert_not_success(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes['lazyllm.status'] == 'error'
    assert spans[0].status.status_code.name == 'ERROR'


@pytest.mark.parametrize('kind', ['module', 'callable', 'generator', 'async_generator'])
def test_cancelled_execution_is_not_traced_as_success(exporter, kind):
    cancelled = asyncio.CancelledError('cancelled by user')

    def stop():
        raise cancelled

    def stream():
        yield 'ready'
        stop()

    async def astream():
        yield 'ready'
        stop()

    class CancelledModule(lazyllm.ModuleBase):
        def forward(self):
            stop()

    async def consume():
        iterator = lazyllm.enable_trace(astream)
        assert await anext(iterator) == 'ready'
        await anext(iterator)

    with pytest.raises(asyncio.CancelledError) as caught:
        if kind == 'module':
            CancelledModule()()
        elif kind == 'callable':
            lazyllm.enable_trace(stop)
        elif kind == 'generator':
            list(lazyllm.enable_trace(stream))
        else:
            asyncio.run(consume())
    assert caught.value is cancelled
    assert_not_success(exporter)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_early_stream_close_is_not_traced_as_success(exporter, asynchronous):
    def stream():
        yield 'ready'
        pytest.fail('closed producer must not resume')

    async def astream():
        yield 'ready'
        pytest.fail('closed producer must not resume')

    async def consume():
        iterator = lazyllm.enable_trace(astream)
        assert await anext(iterator) == 'ready'
        await iterator.aclose()

    if asynchronous:
        asyncio.run(consume())
    else:
        iterator = lazyllm.enable_trace(stream)
        assert next(iterator) == 'ready'
        iterator.close()
    assert_not_success(exporter)
