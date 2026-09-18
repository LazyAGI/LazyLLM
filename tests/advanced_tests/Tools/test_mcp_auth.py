import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from lazyllm.tools.mcp.client import MCPClient


def unauthorized():
    response = httpx.Response(401, request=httpx.Request('POST', 'https://mcp.example'))
    return httpx.HTTPStatusError('unauthorized', request=response.request, response=response)


@pytest.mark.parametrize('operation', ['list_tools', 'call_tool'])
def test_recovers_explicit_401_once(operation):
    attempts = []
    recoveries = []

    async def recover():
        recoveries.append(True)
        return True

    client = MCPClient('https://mcp.example', auth_recovery=recover)

    @asynccontextmanager
    async def session():
        class Session:
            async def list_tools(self):
                attempts.append(True)
                if len(attempts) == 1:
                    raise ExceptionGroup('transport', [unauthorized()])
                return 'ok'

            async def call_tool(self, *args):
                return await self.list_tools()
        yield Session()

    client._run_session = session
    args = ('write', {}) if operation == 'call_tool' else ()
    assert asyncio.run(getattr(client, operation)(*args)) == 'ok'
    assert len(attempts) == 2
    assert len(recoveries) == 1


@pytest.mark.parametrize('error', [TimeoutError('timeout'), RuntimeError('tool error'),
                                  ExceptionGroup('mixed', [unauthorized(), TimeoutError()])])
def test_does_not_replay_ambiguous_failures(error):
    async def recover():
        pytest.fail('must not recover non-401 failures')
    client = MCPClient('https://mcp.example', auth_recovery=recover)
    @asynccontextmanager
    async def session():
        raise error
        yield
    client._run_session = session
    with pytest.raises(type(error)):
        asyncio.run(client.call_tool('write', {}))


def test_persistent_401_stops_after_one_recovery():
    recoveries = []
    async def recover():
        recoveries.append(True)
        return True
    client = MCPClient('https://mcp.example', auth_recovery=recover)
    @asynccontextmanager
    async def session():
        raise unauthorized()
        yield
    client._run_session = session
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.list_tools())
    assert len(recoveries) == 1


def test_real_http_session_refreshes_headers_and_refuses_redirects():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []
    mode = {'redirect': False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, self.headers.get('Authorization'), body['method']))
            if mode['redirect']:
                self.send_response(307)
                self.send_header('Location', '/leak')
                self.end_headers()
                return
            if self.headers.get('Authorization') != 'Bearer fresh':
                self.send_response(401)
                self.end_headers()
                return
            if 'id' not in body:
                self.send_response(202)
                self.end_headers()
                return
            result = ({'protocolVersion': '2025-03-26', 'capabilities': {},
                       'serverInfo': {'name': 'test', 'version': '1'}}
                      if body['method'] == 'initialize' else {'tools': []})
            data = json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': result}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = {'value': 'stale'}
    async def provider():
        return {'Authorization': 'Bearer ' + token['value']}
    async def recover():
        token['value'] = 'fresh'
        return True
    client = MCPClient(f'http://127.0.0.1:{server.server_port}/mcp',
                       auth_provider=provider, auth_recovery=recover)
    try:
        assert asyncio.run(client.list_tools()).tools == []
        assert requests[0][1] == 'Bearer stale'
        assert requests[-1][1:] == ('Bearer fresh', 'tools/list')
        mode['redirect'] = True
        with pytest.raises(Exception):
            asyncio.run(client.list_tools())
        assert all(path == '/mcp' for path, _, _ in requests)
    finally:
        server.shutdown()
        server.server_close()


def test_session_cleanup_401_never_replays_completed_tool():
    executed = []
    async def recover():
        pytest.fail('cleanup must not replay completed tool')
    client = MCPClient('https://mcp.example', auth_recovery=recover)
    @asynccontextmanager
    async def session():
        class Session:
            async def call_tool(self, *args):
                executed.append(True)
                return 'written'
        yield Session()
        raise unauthorized()
    client._run_session = session
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.call_tool('write', {}))
    assert executed == [True]
