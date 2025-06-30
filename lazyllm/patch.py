import requests
from urllib.parse import urlparse
import ipaddress

def _is_ip_address_url(url: str) -> bool:
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return False
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False

def patch_request_func(fname):
    _old_func = getattr(requests, fname)

    def new_func(url, **kwargs):
        if _is_ip_address_url(url):
            try:
                kw = kwargs.copy()
                kw['proxies'] = kw.get('proxies', {'http': None, 'https': None})
                return _old_func(url, **kw)
            except Exception:
                if kwargs.get('proxies'): raise
        return _old_func(url, **kwargs)

    setattr(requests, fname, new_func)

for fname in ['get', 'post', 'delete', 'put', 'patch', 'head']:
    patch_request_func(fname)
