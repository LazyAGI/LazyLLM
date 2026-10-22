import asyncio
import errno
import copy
import hashlib
import json
import mimetypes
import os
import socket
import ssl
import select
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3 import exceptions as urllib3_errors
from urllib3.util.connection import allowed_gai_family
from urllib3.util.ssl_ import create_urllib3_context
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

from lazyllm.common.threading import Thread
from lazyllm.thirdparty import socks
from .model_outcome import ModelCallTerminal
from ....module import ModuleExecutionError


@dataclass(frozen=True)
class ModelCallControl:
    deadline: Optional[float] = None
    cancel_check: Optional[Callable[[], bool]] = field(default=None, repr=False)

    def check(self):
        if self.cancel_check is not None:
            try:
                cancelled = self.cancel_check()
            except Exception as exc:
                raise ModuleExecutionError('Model call cancellation check failed.') from exc
            if cancelled:
                raise asyncio.CancelledError('Model call cancelled.')
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise requests.Timeout('Model call deadline exceeded.')

    def timeout(self, configured):
        self.check()
        if self.deadline is None:
            return configured
        remaining = max(0.001, self.deadline - time.monotonic())
        if isinstance(configured, (list, tuple)):
            return tuple(min(float(value), remaining) if value is not None else remaining for value in configured)
        return min(float(configured), remaining) if configured is not None else remaining

    def wait(self, duration):
        until = time.monotonic() + duration
        while True:
            self.check()
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))


@dataclass(frozen=True)
class ModelCallResult:
    value: Any = field(repr=False)
    terminal: ModelCallTerminal

    @property
    def usage(self):
        return copy.deepcopy(self.terminal.usage or {})


def _requires_vision(value, files):
    def contains_image(item):
        if isinstance(item, dict):
            return item.get('type') in {'image_url', 'input_image', 'image'} or any(
                contains_image(child) for child in item.values())
        return isinstance(item, (list, tuple)) and any(contains_image(child) for child in item)
    return contains_image(value) or any(
        (mimetypes.guess_type(str(file).split('?', 1)[0])[0] or '').startswith('image/') for file in files or [])


@dataclass(frozen=True)
class PreparedModelRequest:
    _module: Any = field(repr=False, compare=False)
    _json: str = field(repr=False)
    _url: str = field(repr=False)
    _stream: Any = field(repr=False, compare=False)
    _timeout: Any = field(repr=False)
    _max_attempts: int = field(repr=False)
    requires_vision: bool = False
    _caller: Any = field(default=None, repr=False, compare=False)
    _args: tuple = field(default=(), repr=False, compare=False)
    _kwargs: dict = field(default_factory=dict, repr=False, compare=False)
    _control: Any = field(default=None, repr=False, compare=False)
    is_estimate: bool = field(default=True, init=False)

    def __repr__(self):
        digest = hashlib.sha256((self._url + '\0' + self._json).encode()).hexdigest()
        return f'PreparedModelRequest(snapshot={digest})'

    @property
    def payload(self):
        return json.loads(self._json)

    @property
    def model_type(self):
        return self._module.type

    def estimate_tokens(self, text_estimator, message_estimator=None):
        data = self.payload
        messages = data.pop('messages', [])
        estimate_message = message_estimator or (
            lambda message: text_estimator(json.dumps(message, ensure_ascii=False)))
        return text_estimator(json.dumps({**data, 'messages': []}, ensure_ascii=False)) + sum(
            estimate_message(message) + 4 for message in messages)

    def execute(self, *, control=None):
        caller = self._caller or self._module
        args, kwargs = copy.deepcopy((self._args, self._kwargs))
        request = replace(self, _control=control)
        result = None

        def call(*_args, **_kwargs):
            nonlocal result
            if control is not None: control.check()
            # Cache identity comes from the frozen body/URL. Per-call control and
            # original file paths must not alter it after the snapshot is made.
            result = caller._call_impl(prepared_request=request, return_model_result=True)
            if control is not None: control.check()
            return result.value

        value = caller._call_with_hooks(args, kwargs, call)
        return ModelCallResult(value, copy.deepcopy(result.terminal))


# DNS work never carries a prompt or credential. Pending lookups are cancelled;
# in-progress OS resolution may finish later, without creating an HTTP request.
_resolver = ThreadPoolExecutor(max_workers=4, thread_name_prefix='lazyllm-dns')


def _resolve_addresses(host, port, check, family=None):
    future = _resolver.submit(socket.getaddrinfo, host, port,
                              allowed_gai_family() if family is None else family, socket.SOCK_STREAM)
    try:
        while not future.done():
            check()
            time.sleep(0.01)
        check()
        return future.result()
    finally:
        future.cancel()


def _connect_socks(sock, target, connection, control, check):
    options = connection._socks_options
    host = connection.host.strip('[]')
    if not options['rdns']:
        family = socket.AF_INET if options['socks_version'] == socks.SOCKS4 else socket.AF_UNSPEC
        host = _resolve_addresses(host, connection.port, check, family)[0][4][0]
    sock.set_proxy(options['socks_version'], target[0], target[1], options['rdns'],
                   options['username'], options['password'])
    sock.settimeout(control.timeout(connection.timeout))
    check()
    try:
        # PySocks owns authentication and negotiation; the guard can interrupt
        # this socket before connect() returns, including a stalled handshake.
        sock.connect((host, connection.port))
    except socks.ProxyError as exc:
        if exc.socket_err is not None:
            raise exc.socket_err from exc
        raise


def _connect_address(address, connection, control, check, sockets):
    family, kind, protocol, _, target = address
    is_socks = hasattr(connection, '_socks_options')
    sock = (socks.socksocket if is_socks else socket.socket)(family, kind, protocol)
    sockets.append(sock)
    try:
        for option in connection.socket_options or []:
            sock.setsockopt(*option)
        if connection.source_address:
            sock.bind(connection.source_address)
        if is_socks:
            _connect_socks(sock, target, connection, control, check)
            check()
            return sock
        sock.setblocking(False)
        result = sock.connect_ex(target)
        if result not in {0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY}:
            raise OSError(result, 'Connection failed.')
        while result:
            check()
            _, writable, exceptional = select.select([], [sock], [sock], 0.02)
            if writable or exceptional:
                result = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if result:
                    raise OSError(result, 'Connection failed.')
        check()
        sock.settimeout(control.timeout(connection.timeout))
        return sock
    except BaseException:
        sock.close()
        raise


def _controlled_socket(connection, guard):
    timeout = guard.control.timeout(connection.timeout)
    deadline = time.monotonic() + timeout if timeout is not None else None

    def check():
        guard.check()
        if deadline is not None and time.monotonic() >= deadline:
            raise socket.timeout('Connection timed out.')

    try:
        options = getattr(connection, '_socks_options', None)
        host, port = ((options['proxy_host'], options['proxy_port'] or 1080) if options else
                      (connection._dns_host, connection.port))
        addresses = _resolve_addresses(host.strip('[]'), port, check)
        error = None
        for address in addresses:
            check()
            try:
                sock = _connect_address(address, connection, guard.control, check, guard.sockets)
                sys.audit('http.client.connect', connection, connection.host, connection.port)
                return sock
            except OSError as exc:
                error = exc
        raise error or OSError('No connection address is available.')
    except socket.gaierror as exc:
        if hasattr(urllib3_errors, 'NameResolutionError'):
            raise urllib3_errors.NameResolutionError(connection.host, connection, exc) from exc
        raise NewConnectionError(connection, 'Name resolution failed.') from exc
    except socket.timeout as exc:
        raise ConnectTimeoutError(connection, 'Connection timed out.') from exc
    except OSError as exc:
        raise NewConnectionError(connection, 'Connection failed.') from exc


class _RequestGuard:
    def __init__(self, control):
        self.control = control
        self.stopped = threading.Event()
        self.connections, self.sockets = [], []
        self.failure = None
        self.watcher = Thread(target=self.watch, daemon=True, local_scope='inherit')

    def check(self):
        if self.failure is not None:
            raise self.failure
        self.control.check()

    def watch(self):
        while not self.stopped.wait(0.02):
            try:
                self.check()
            except BaseException as exc:  # noqa: B036 - re-raised on the request thread by check().
                self.failure = exc
                for sock in [*self.sockets, *(conn.sock for conn in self.connections if conn.sock)]:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                # A not-yet-connected socket can survive shutdown on macOS.
                # Keep interrupting until the request thread has exited.

    def tls_context(self, verify=None, *, proxy=False):
        context = create_urllib3_context()
        if proxy:
            context.check_hostname = True
        if verify is True:
            context.load_verify_locations(cafile=requests.certs.where())
        elif verify is False:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif isinstance(verify, str):
            context.load_verify_locations(**{'capath' if os.path.isdir(verify) else 'cafile': verify})
        original = context.wrap_socket
        guard = self

        def wrap_socket(sock, *args, **params):
            handshake = params.pop('do_handshake_on_connect', True)
            wrapped = original(sock, *args, do_handshake_on_connect=False, **params)
            guard.sockets.append(wrapped)
            try:
                guard.check()
                if handshake:
                    wrapped.do_handshake()
                guard.check()
                return wrapped
            except BaseException:
                wrapped.close()
                raise

        context.wrap_socket = wrap_socket
        return context

    def adapter(self):
        guard = self

        class TrackedConnection:
            def __init__(self, *args, **params):
                super().__init__(*args, **params)
                guard.connections.append(self)

            def _new_conn(self):
                return _controlled_socket(self, guard)

        def track_pools(manager):
            pools = {}
            for scheme, pool in manager.pool_classes_by_scheme.items():
                class Connection(TrackedConnection, pool.ConnectionCls):
                    pass

                class Pool(pool):
                    ConnectionCls = Connection

                pools[scheme] = Pool
            manager.pool_classes_by_scheme = pools

        class Adapter(HTTPAdapter):
            def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
                self._proxy_verify = verify
                pool = super().get_connection_with_tls_context(request, verify, proxies, cert)
                if pool.scheme == 'https':
                    # Requests 2.32.2 does not have build_connection_pool_key_attributes().
                    # Use its shared extension point without mutating the global TLS context.
                    pool.conn_kw['ssl_context'] = guard.tls_context(verify)
                return pool

            def init_poolmanager(self, *args, **params):
                params.setdefault('ssl_context', guard.tls_context())
                super().init_poolmanager(*args, **params)
                track_pools(self.poolmanager)

            def proxy_manager_for(self, proxy, **params):
                if proxy in self.proxy_manager:
                    return self.proxy_manager[proxy]
                params.setdefault('ssl_context', guard.tls_context())
                if proxy.lower().startswith('https:'):
                    params.setdefault('proxy_ssl_context', guard.tls_context(self._proxy_verify, proxy=True))
                manager = super().proxy_manager_for(proxy, **params)
                track_pools(manager)
                return manager

        return Adapter()


@contextmanager
def model_response(url, *, control=None, **kwargs):
    if control is None:
        with requests.post(url, **kwargs) as response:
            yield response
        return
    guard = _RequestGuard(control)
    guard.check()
    guard.watcher.start()
    try:
        with requests.Session() as session:
            session.mount('http://', guard.adapter())
            session.mount('https://', guard.adapter())
            kwargs['timeout'] = control.timeout(kwargs.get('timeout'))
            with session.post(url, **kwargs) as response:
                guard.check()
                yield response
                guard.check()
    except BaseException:
        guard.check()
        raise
    finally:
        guard.stopped.set()
        guard.watcher.join()
