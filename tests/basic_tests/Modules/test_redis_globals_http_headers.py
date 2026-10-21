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


pytestmark = pytest.mark.skipif(
    not os.environ.get('LAZYLLM_REDIS_DELETE_TEST_URL'),
    reason='Requires an explicitly supplied isolated Redis via LAZYLLM_REDIS_URL',
)


@pytest.fixture(scope='module')
def relay_url(tmp_path_factory):
    class Application:
        def __call__(self, query):
            return self.read_state(query)

        def read_state(self, query):
            return lazyllm.globals['redis_header_regression'] + ':' + query

    application = Application()

    root = tmp_path_factory.mktemp('native-redis-header-relay')
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


@pytest.mark.parametrize('method', ['forward', '_call'])
def test_redis_globals_cross_real_relay_http_headers(relay_url, method):
    lazyllm.globals['redis_header_regression'] = 'session-value'
    module = lazyllm.ServerModule(url=relay_url)
    try:
        result = module('query') if method == 'forward' else module._call('read_state', 'query')
        assert result == 'session-value:query'
    finally:
        lazyllm.globals.pop('redis_header_regression', None)
