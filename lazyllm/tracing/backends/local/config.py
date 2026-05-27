from pathlib import Path

from lazyllm.common import LOG
from lazyllm.configs import config


_DEFAULT_STORAGE_DIR = Path(config['home']) / 'traces'
_DEFAULT_ARCHIVE_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_ARCHIVE_RETENTION_SECONDS = 30 * 24 * 60 * 60

config.add(
    'trace_local_storage_dir',
    str,
    str(_DEFAULT_STORAGE_DIR),
    'TRACE_LOCAL_STORAGE_DIR',
    description='Directory used by the local tracing backend to store JSONL trace files.',
)
config.add(
    'trace_local_archive_seconds',
    int,
    _DEFAULT_ARCHIVE_SECONDS,
    'TRACE_LOCAL_ARCHIVE_SECONDS',
    description='Seconds before local JSONL trace files are archived into ZIP files.',
)
config.add(
    'trace_local_archive_retention_seconds',
    int,
    _DEFAULT_ARCHIVE_RETENTION_SECONDS,
    'TRACE_LOCAL_ARCHIVE_RETENTION_SECONDS',
    description='Seconds before local trace ZIP archives are deleted.',
)


def read_local_storage_dir() -> Path:
    value = config['trace_local_storage_dir']
    if not value:
        LOG.warning(f'LAZYLLM_TRACE_LOCAL_STORAGE_DIR is empty; using {_DEFAULT_STORAGE_DIR}')
        value = str(_DEFAULT_STORAGE_DIR)
    return Path(value).expanduser().resolve()


def read_local_archive_seconds() -> int:
    return config['trace_local_archive_seconds']


def read_local_archive_retention_seconds() -> int:
    return config['trace_local_archive_retention_seconds']


__all__ = [
    'read_local_archive_retention_seconds',
    'read_local_archive_seconds',
    'read_local_storage_dir',
]
