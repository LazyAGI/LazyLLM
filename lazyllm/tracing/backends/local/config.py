from pathlib import Path

from lazyllm.common import LOG
from lazyllm.configs import config


_DEFAULT_STORAGE_DIR = Path(config['home']) / 'traces'

config.add(
    'trace_local_storage_dir',
    str,
    str(_DEFAULT_STORAGE_DIR),
    'TRACE_LOCAL_STORAGE_DIR',
    description='Directory used by the local tracing backend to store JSONL trace files.',
)


def read_local_storage_dir() -> Path:
    value = config['trace_local_storage_dir']
    if not value:
        LOG.warning(f'LAZYLLM_TRACE_LOCAL_STORAGE_DIR is empty; using {_DEFAULT_STORAGE_DIR}')
        value = str(_DEFAULT_STORAGE_DIR)
    return Path(value).expanduser().resolve()


__all__ = ['read_local_storage_dir']
