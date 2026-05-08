import os
from pathlib import Path


_DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent / '.temp'


def read_local_storage_dir() -> Path:
    value = os.getenv('LAZYLLM_TRACE_LOCAL_STORAGE_DIR')
    return Path(value).expanduser() if value else _DEFAULT_STORAGE_DIR


__all__ = ['read_local_storage_dir']
