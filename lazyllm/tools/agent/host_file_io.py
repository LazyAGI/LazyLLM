"""Execution-time host-file guard hooks kept separate from prepare metadata."""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from contextvars import ContextVar

from .tool_runtime import resolve_host_path as _resolve_host_path


_HOST_ACCESS_GUARD = ContextVar('host_file_access_guard', default=None)


def resolve_host_path(path):
    return _resolve_host_path(path)


@contextmanager
def host_file_execution_scope(guard):
    token = _HOST_ACCESS_GUARD.set(guard)
    try:
        yield
    finally:
        _HOST_ACCESS_GUARD.reset(token)


def check_path(path, operation='read'):
    guard = _HOST_ACCESS_GUARD.get()
    return guard.check_path(path, operation) if guard is not None else path


def open_read(path):
    guard = _HOST_ACCESS_GUARD.get()
    if guard is not None:
        return guard.open_read(path)
    return os.fdopen(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb')


def open_write(path, mode='w'):
    guard = _HOST_ACCESS_GUARD.get()
    if guard is not None:
        return guard.open_write(path, mode)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    flags |= os.O_APPEND if mode == 'a' else os.O_EXCL if mode == 'x' else os.O_TRUNC
    return os.fdopen(os.open(path, flags, 0o600), mode + 'b')


def makedirs(path, exist_ok=True, parents=True):
    guard = _HOST_ACCESS_GUARD.get()
    if guard is not None:
        return guard.makedirs(path, exist_ok=exist_ok, parents=parents)
    if parents:
        return os.makedirs(path, exist_ok=exist_ok)
    try:
        return os.mkdir(path)
    except FileExistsError:
        if not exist_ok or not os.path.isdir(path):
            raise


def delete(path, recursive=False):
    guard = _HOST_ACCESS_GUARD.get()
    if guard is not None:
        return guard.delete(path, recursive=recursive)
    if recursive and os.path.isdir(path) and not os.path.islink(path):
        return shutil.rmtree(path)
    return os.rmdir(path) if os.path.isdir(path) and not os.path.islink(path) else os.unlink(path)


def rename(src, dst, overwrite=False):
    guard = _HOST_ACCESS_GUARD.get()
    if guard is not None:
        return guard.rename(src, dst, overwrite=overwrite)
    if not overwrite and os.path.lexists(dst):
        raise FileExistsError(dst)
    return shutil.move(src, dst)


def copy_file(src, dst, overwrite=False):
    guard = _HOST_ACCESS_GUARD.get()
    if guard is not None:
        return guard.copy(src, dst, overwrite=overwrite)
    with open_read(src) as source:
        with open_write(dst, 'w' if overwrite else 'x') as target:
            shutil.copyfileobj(source, target)
    return dst


def walk(path):
    guard = _HOST_ACCESS_GUARD.get()
    return guard.walk(path) if guard is not None else os.walk(path, followlinks=False)


def listdir(path):
    guard = _HOST_ACCESS_GUARD.get()
    return guard.listdir(path) if guard is not None else os.listdir(path)
