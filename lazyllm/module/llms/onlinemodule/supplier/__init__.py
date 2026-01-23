import importlib
import pkgutil

_package = __name__
_import_errors = {}

for _mod in pkgutil.iter_modules(__path__):
    name = _mod.name
    if name.startswith('_'):
        continue
    try:
        importlib.import_module(f'{_package}.{name}')
    except Exception as exc:  # keep import best-effort to avoid aborting auto-registration
        _import_errors[name] = exc
