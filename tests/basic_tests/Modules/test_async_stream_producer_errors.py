import asyncio
import base64
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
import requests

import lazyllm
from lazyllm.module.module import ModuleExecutionError


@pytest.fixture(scope='module')
def relay_url(tmp_path_factory):
    def application(query, **kwargs):
        if query == 'before':
            raise ValueError('native stream producer failed before output')
        if query == 'during':
            def chunks():
                yield 'partial'
                raise ValueError('native stream producer failed during output')
            return chunks()
        return 'accepted ' + query

    root = tmp_path_factory.mktemp('native-stream-relay')
    payload = root / 'application.pkl'
    payload.write_bytes(base64.b64decode(lazyllm.dump_obj(application)))
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    server = Path(lazyllm.__file__).parent / 'components/deploy/relay/server.py'
    with (root / 'relay.log').open('w') as log:
        process = subprocess.Popen([
            sys.executable, str(server), '--open_ip=127.0.0.1', '--open_port=' + str(port),
            '--function=@file:' + str(payload), '--pythonpath=' + str(Path(lazyllm.__file__).parent.parent),
        ], env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                assert process.poll() is None, (root / 'relay.log').read_text()
                try:
                    if requests.get(url + '/docs', timeout=0.5).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.05)
            else:
                pytest.fail('Native Relay did not become ready')
            yield url + '/generate'
        finally:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.parametrize('query', ['before', 'during'])
def test_public_async_stream_propagates_real_relay_failure(relay_url, query):
    module = lazyllm.ServerModule(url=relay_url, stream=True)

    async def consume():
        return [chunk async for chunk in module.astream_call(query)]

    with pytest.raises(ModuleExecutionError):
        asyncio.run(consume())


def test_public_async_stream_normal_completion(relay_url):
    module = lazyllm.ServerModule(url=relay_url, stream=True)

    async def consume():
        return [chunk async for chunk in module.astream_call('sample')]

    assert asyncio.run(consume()) == ['accepted sample']
