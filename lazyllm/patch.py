import os
import sys
import ipaddress
import importlib.abc
import importlib.util
from typing import Callable
from urllib.parse import urlparse

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
    import requests
    with requests.sessions.Session() as session:
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                session.trust_env = False
                return session.request(method=method, url=url, **kwargs)
            except Exception: pass
        session.trust_env = True
        return session.request(method=method, url=url, **kwargs)

def _get(url, params=None, **kwargs): return request('get', url, params=params, **kwargs)
def _options(url, **kwargs): return request('options', url, **kwargs)
def _post(url, data=None, json=None, **kwargs): return request('post', url, data=data, json=json, **kwargs)
def _put(url, data=None, **kwargs): return request('put', url, data=data, **kwargs)
def _patch(url, data=None, **kwargs): return request('patch', url, data=data, **kwargs)
def _delete(url, **kwargs): return request('delete', url, **kwargs)

def _head(url, **kwargs):
    kwargs.setdefault('allow_redirects', False)
    return request('head', url, **kwargs)

def patch_httpx_func(httpx, fname):
    _old_func = getattr(httpx, fname)

    def new_func(url, **kwargs):
        if os.environ.get('http_proxy') and _is_ip_address_url(url):
            try:
                return _old_func(url, **{**kwargs, **dict(trust_env=False)})
            except Exception: pass
        return _old_func(url, **kwargs)

    setattr(httpx, fname, new_func)

def patch_requests_and_httpx():  # noqa: C901
    import requests
    import httpx

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

    for fname in ['get', 'options', 'post', 'delete', 'put', 'patch', 'head']:
        patch_httpx_func(httpx, fname)


class LazyPatchLoader(importlib.abc.Loader):
    """Lazy Patch Loader for automatically applying patches during module loading.

The ``LazyPatchLoader`` is an import system loader that automatically applies patches 
to the requests and httpx libraries when a module is executed. This loader wraps the 
original module specification and automatically calls patch functions after module execution.

Args:
    original_spec (ModuleSpec): The original module's specification object containing 
                                module loading information and paths.

Features:

- Automatically sets correct package and path attributes during module loading
- Executes the original loader's module execution logic
- Automatically applies patches to requests and httpx libraries after module execution

"""
    def __init__(self, original_spec):
        self.original_spec = original_spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        """Execute the module loading and initialization process.

This method is the core method of the import system loader, responsible for executing 
the module's code and initializing the module object. In LazyPatchLoader, this method 
first sets the module's package and path attributes, then executes the original loader's 
module execution logic, and finally automatically applies patches to the requests and 
httpx libraries.

Args:
    module (ModuleType): The module object to be executed.

"""
        if self.original_spec.submodule_search_locations is not None:
            module.__package__ = self.original_spec.name
        elif '.' in self.original_spec.name:
            module.__package__ = self.original_spec.name.rpartition('.')[0]
        else:
            module.__package__ = ''

        if self.original_spec.submodule_search_locations is not None:
            module.__path__ = self.original_spec.submodule_search_locations

        self.original_spec.loader.exec_module(module)
        patch_requests_and_httpx()

class LazyPatchFinder(importlib.abc.MetaPathFinder):
    """Lazy Patch Finder for intercepting specific module imports and applying patches.

The ``LazyPatchFinder`` is a meta path finder that intercepts import requests for 
'requests' and 'httpx' modules during the import process, and uses a custom 
LazyPatchLoader to load these modules, automatically applying patches during module loading.

**Note:**

- This finder only takes effect when modules are not already imported
- If modules are already imported, directly calls patch_requests_and_httpx() function
- Maintains the original attributes and paths of the modules
"""
    def find_spec(self, fullname, path, target=None):
        """Find and return the module specification object for custom module loading process.

This method is the core method of MetaPathFinder, responsible for finding the specification 
object of the specified module during the import process. In LazyPatchFinder, it specifically 
intercepts import requests for 'requests' and 'httpx' modules, using a custom LazyPatchLoader 
to wrap the original module specification.

Args:
    fullname (str): The full name of the module to import
    path (list): Search path list, None for top-level modules
    target (module, optional): Target module object (used during reloading)

**Returns:**

- For 'requests' and 'httpx' modules: Returns module specification wrapped with LazyPatchLoader
- For other modules: Returns None, allowing other finders to continue processing

"""
        if fullname in ['requests', 'httpx']:
            if self in sys.meta_path: sys.meta_path.remove(self)
            original_spec = importlib.util.find_spec(fullname)
            if original_spec is None: return None
            return importlib.util.spec_from_loader(fullname, LazyPatchLoader(original_spec),
                                                   origin=original_spec.origin)
        return None

if 'httpx' in sys.modules or 'requests' in sys.modules:
    patch_requests_and_httpx()
else:
    sys.meta_path.insert(0, LazyPatchFinder())


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
