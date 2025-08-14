import requests
import httpx
from urllib.parse import urlparse
import ipaddress
import os
from typing import Callable

def _is_ip_address_url(url: str) -> bool:
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return False
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False

no_proxies = set(os.environ.get('no_proxy', '').split(','))
no_proxies.update({'localhost', '127.0.0.1', 'localaddress', '.localdomain.com'})
os.environ['no_proxy'] = ','.join(list(no_proxies))


def request(method, url, **kwargs):
    with requests.sessions.Session() as session:
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                session.trust_env = False
                return session.request(method=method, url=url, **kwargs)
            except Exception: pass
        session.trust_env = True
        return session.request(method=method, url=url, **kwargs)


def _get(url, params=None, **kwargs): return request("get", url, params=params, **kwargs)
def _options(url, **kwargs): return request("options", url, **kwargs)
def _post(url, data=None, json=None, **kwargs): return request("post", url, data=data, json=json, **kwargs)
def _put(url, data=None, **kwargs): return request("put", url, data=data, **kwargs)
def _patch(url, data=None, **kwargs): return request("patch", url, data=data, **kwargs)
def _delete(url, **kwargs): return request("delete", url, **kwargs)
def _head(url, **kwargs):
    kwargs.setdefault("allow_redirects", False)
    return request("head", url, **kwargs)


requests.get, requests.options, requests.post = _get, _options, _post
requests.put, requests.patch, requests.delete, requests.head = _put, _patch, _delete, _head


_old_httpx_func = httpx.request

def new_httpx_func(method, url, **kwargs):
    if os.environ.get('http_proxy') and _is_ip_address_url(url):
        try:
            return _old_httpx_func(method, url, **{**kwargs, **dict(trust_env=False)})
        except Exception: pass
    return _old_httpx_func(method, url, **kwargs)

httpx.request = new_httpx_func


def patch_httpx_func(fname):
    _old_func = getattr(httpx, fname)

    def new_func(url, **kwargs):
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                return _old_func(url, **{**kwargs, **dict(trust_env=False)})
            except Exception: pass
        return _old_func(url, **kwargs)

    setattr(httpx, fname, new_func)

for fname in ['get', 'options', 'post', 'delete', 'put', 'patch', 'head']:
    patch_httpx_func(fname)


def patch_os_env(set_action: Callable[[str, str], None], unset_action: Callable[[str], None]):

    old_setitem = os._Environ.__setitem__

    def new_setitem(self, key, value):
        old_setitem(self, key, value)
        if isinstance(key, bytes): key = key.decode('utf-8')
        if key.lower().startswith('lazyllm_'): set_action(key, value)

    old_delitem = os._Environ.__delitem__

    def new_delitem(self, key):
        old_delitem(self, key)
        if isinstance(key, bytes): key = key.decode('utf-8')
        if key.lower().startswith('lazyllm_'): unset_action(key)

    os._Environ.__setitem__ = new_setitem
    os._Environ.__delitem__ = new_delitem
