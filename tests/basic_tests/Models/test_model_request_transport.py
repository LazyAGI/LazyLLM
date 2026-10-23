import asyncio
import os
import select
import socket
import socketserver
import ssl
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
import trustme

from lazyllm.module.llms.onlinemodule.base.model_request import ModelCallControl, model_response, response_chunks


@contextmanager
def serving(server):
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize('controlled', [False, True])
def test_truncated_response_is_a_transport_failure_even_after_a_terminal_frame(controlled):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"stop"}]}\n\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body) + 100))
            self.end_headers()
            self.wfile.write(body)

    with serving(ThreadingHTTPServer(('127.0.0.1', 0), Handler)) as server:
        control = ModelCallControl() if controlled else None
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            with model_response(f'http://127.0.0.1:{server.server_port}/chat',
                                json={'model': 'fixture'}, stream=True, control=control) as response:
                list(response.iter_lines())


@pytest.fixture
def proxy_endpoint(tmp_path, monkeypatch):  # noqa: C901 - local HTTP, CONNECT and SOCKS protocol fixture.
    ca = trustme.CA()
    certificate = ca.issue_cert('localhost', '127.0.0.1')
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    certificate.configure_cert(context)
    ca_path = tmp_path / 'ca.pem'
    ca.cert_pem.write_to_path(ca_path)
    monkeypatch.setenv('REQUESTS_CA_BUNDLE', str(ca_path))
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('NO_PROXY', '')
    seen, destinations, credentials = [], [], []
    reached, release, disconnected = (threading.Event() for _ in range(3))

    @contextmanager
    def start(scheme, target_scheme, phase):
        class Origin(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                seen.append(self.rfile.read(int(self.headers['Content-Length'])))
                reached.set()
                if phase == 'headers':
                    release.wait(3)
                try:
                    self.send_response(200)
                    self.end_headers()
                    if phase == 'stream':
                        self.wfile.write(b'data: partial\n\n')
                        self.wfile.flush()
                        release.wait(3)
                    else:
                        self.wfile.write(b'proxied')
                except (OSError, ssl.SSLError):
                    pass

        origin = ThreadingHTTPServer(('127.0.0.1', 0), Origin)
        if target_scheme == 'https':
            origin.socket = context.wrap_socket(origin.socket, server_side=True)

        class Proxy(socketserver.StreamRequestHandler):
            rbufsize = 0

            def handle(self):
                self.connection.settimeout(3)
                try:
                    self.forward()
                except (OSError, EOFError):
                    pass
                finally:
                    disconnected.set()

            def read(self, count):
                data = b''
                while len(data) < count:
                    chunk = self.rfile.read(count - len(data))
                    if not chunk:
                        raise EOFError()
                    data += chunk
                return data

            def terminated(self):
                data = b''
                while (byte := self.read(1)) != b'\0':
                    data += byte
                return data

            def forward(self):
                first = self.read(1)
                if phase == 'handshake':
                    reached.set()
                    # Observe that cancellation closes the connection while negotiation is blocked.
                    while self.connection.recv(4096):
                        pass
                    return
                prefix = b''
                if first == b'\x05':
                    methods = self.read(self.read(1)[0])
                    assert 2 in methods
                    self.wfile.write(b'\x05\x02')
                    assert self.read(1) == b'\x01'
                    user = self.read(self.read(1)[0])
                    password = self.read(self.read(1)[0])
                    credentials.append((user, password))
                    self.wfile.write(b'\x01\x00')
                    assert self.read(3) == b'\x05\x01\x00'
                    kind = self.read(1)[0]
                    if kind == 3:
                        host = self.read(self.read(1)[0]).decode()
                    else:
                        family, count = (socket.AF_INET, 4) if kind == 1 else (socket.AF_INET6, 16)
                        host = socket.inet_ntop(family, self.read(count))
                    port = int.from_bytes(self.read(2), 'big')
                    self.wfile.write(b'\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00')
                elif first == b'\x04':
                    assert self.read(1) == b'\x01'
                    port = int.from_bytes(self.read(2), 'big')
                    address = self.read(4)
                    credentials.append((self.terminated(), b''))
                    host = self.terminated().decode() if address == b'\0\0\0\x01' else socket.inet_ntoa(address)
                    self.wfile.write(b'\x00\x5a' + port.to_bytes(2, 'big') + b'\x7f\x00\x00\x01')
                else:
                    line = first + self.rfile.readline()
                    headers = b''
                    while (header := self.rfile.readline()) != b'\r\n':
                        if not header:
                            raise EOFError()
                        headers += header
                    credentials.append(headers)
                    host, port = 'localhost', origin.server_port
                    if line.startswith(b'CONNECT '):
                        self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
                    else:
                        prefix = line + headers + b'\r\n'
                destinations.append((host, port))
                with socket.create_connection(('127.0.0.1', origin.server_port), timeout=3) as upstream:
                    if prefix:
                        upstream.sendall(prefix)
                    while not release.is_set():
                        ready, _, _ = select.select([self.connection, upstream], [], [], 0.02)
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            (upstream if source is self.connection else self.connection).sendall(data)

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True

        proxy = Server(('127.0.0.1', 0), Proxy)
        if scheme == 'https':
            proxy.socket = context.wrap_socket(proxy.socket, server_side=True)
        with serving(origin), serving(proxy):
            try:
                yield (f'{scheme}://fixture-user:fixture-pass@localhost:{proxy.server_address[1]}',
                       f'{target_scheme}://localhost:{origin.server_port}/chat')
            finally:
                release.set()

    return start, seen, destinations, credentials, reached, disconnected


@pytest.mark.parametrize('scheme', ['http', 'https', 'socks4', 'socks4a', 'socks5', 'socks5h'])
@pytest.mark.parametrize('target_scheme', ['http', 'https'])
@pytest.mark.parametrize('controlled', [False, True])
def test_controlled_proxy_request_preserves_routing_and_auth(proxy_endpoint, scheme, target_scheme, controlled):
    start, seen, destinations, credentials, _, _ = proxy_endpoint
    with start(scheme, target_scheme, 'normal') as (proxy, target):
        with model_response(target, data=b'fixture-body', proxies={target_scheme: proxy},
                            control=ModelCallControl() if controlled else None,
                            timeout=3) as response:
            assert response.content == b'proxied'
    assert seen == [b'fixture-body']
    assert len(destinations) == 1
    if scheme.startswith('socks'):
        if scheme in ('socks4a', 'socks5h'):
            assert destinations[0][0] == 'localhost'
        else:
            assert destinations[0][0] in ('127.0.0.1', '::1')
        assert credentials == [(b'fixture-user', b'fixture-pass' if scheme.startswith('socks5') else b'')]
    else:
        assert b'Proxy-Authorization: Basic ' in credentials[0]


@pytest.mark.parametrize('scheme', ['http', 'https', 'socks5h'])
@pytest.mark.parametrize('phase', ['handshake', 'headers', 'stream'])
@pytest.mark.parametrize('cancelled', [False, True])
def test_proxy_timeout_and_cancel_after_blocking_io(proxy_endpoint, scheme, phase, cancelled):
    start, seen, _, _, reached, disconnected = proxy_endpoint
    cancel = threading.Event()
    with start(scheme, 'https', phase) as (proxy, target):
        def stop():
            if reached.wait(1):
                cancel.set()

        thread = threading.Thread(target=stop)
        if cancelled:
            thread.start()
        started = time.monotonic()
        control = ModelCallControl(cancel_check=cancel.is_set)
        try:
            with pytest.raises(asyncio.CancelledError if cancelled else requests.RequestException):
                with model_response(target, data=b'fixture-body', proxies={'https': proxy},
                                    control=control, stream=True, timeout=(0.2, 0.2)) as response:
                    list(response_chunks(response, control))
            assert time.monotonic() - started < 1.5
            assert reached.is_set()
            assert disconnected.wait(1)
        finally:
            if cancelled:
                thread.join(2)
    assert len(seen) == (0 if phase == 'handshake' else 1)


def test_dns_failure_remains_a_requests_connection_error(monkeypatch):
    def fail(*_args):
        raise socket.gaierror(socket.EAI_NONAME, 'fixture name resolution failure')

    monkeypatch.setattr(socket, 'getaddrinfo', fail)
    with pytest.raises(requests.ConnectionError):
        with model_response('http://missing.invalid/', control=ModelCallControl(), timeout=1):
            pytest.fail('DNS failure returned a response')


@pytest.mark.parametrize('scheme', ['http', 'https', 'socks5h'])
@pytest.mark.parametrize('verify', [False, True])
def test_proxy_preserves_certificate_verification(proxy_endpoint, monkeypatch, scheme, verify):
    start, seen, _, _, _, _ = proxy_endpoint
    monkeypatch.delenv('REQUESTS_CA_BUNDLE')
    with start(scheme, 'https', 'normal') as (proxy, target):
        def request():
            with model_response(target, data=b'fixture-body', proxies={'https': proxy}, verify=verify,
                                control=ModelCallControl(), timeout=3) as response:
                assert response.content == b'proxied'

        if verify:
            with pytest.raises((requests.exceptions.SSLError, requests.exceptions.ProxyError),
                               match='CERTIFICATE_VERIFY_FAILED'):
                request()
        else:
            request()
    assert seen == ([] if verify else [b'fixture-body'])


@pytest.mark.parametrize('controlled', [False, True])
def test_https_proxy_uses_only_the_selected_ca_bundle(proxy_endpoint, monkeypatch, tmp_path, controlled):
    start, seen, destinations, _, _, _ = proxy_endpoint
    monkeypatch.setenv('SSL_CERT_FILE', os.environ['REQUESTS_CA_BUNDLE'])
    other_ca = tmp_path / 'other-ca.pem'
    trustme.CA().cert_pem.write_to_path(other_ca)
    monkeypatch.setenv('REQUESTS_CA_BUNDLE', str(other_ca))
    with start('https', 'https', 'normal') as (proxy, target):
        with pytest.raises((requests.exceptions.SSLError, requests.exceptions.ProxyError),
                           match='CERTIFICATE_VERIFY_FAILED'):
            with model_response(target, data=b'fixture-body', proxies={'https': proxy}, timeout=1,
                                control=ModelCallControl() if controlled else None):
                pytest.fail('Proxy certificate outside the selected CA bundle was trusted')
    assert seen == destinations == []


@pytest.mark.parametrize('controlled', [False, True])
def test_https_proxy_checks_its_hostname(proxy_endpoint, monkeypatch, controlled):
    start, seen, destinations, _, _, _ = proxy_endpoint
    resolve = socket.getaddrinfo

    def resolve_wrong_name(host, *args):
        return resolve('127.0.0.1' if host == 'wrong.invalid' else host, *args)

    monkeypatch.setattr(socket, 'getaddrinfo', resolve_wrong_name)
    with start('https', 'https', 'normal') as (proxy, target):
        proxy = proxy.replace('localhost', 'wrong.invalid')
        with pytest.raises((requests.exceptions.SSLError, requests.exceptions.ProxyError), match='(?i)hostname'):
            with model_response(target, data=b'fixture-body', proxies={'https': proxy},
                                control=ModelCallControl() if controlled else None, timeout=3):
                pytest.fail('Proxy certificate with a mismatched hostname was trusted')
    assert seen == destinations == []
