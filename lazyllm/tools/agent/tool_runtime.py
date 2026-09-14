import copy
import inspect
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Optional, Tuple


_TOOL_RUNTIME_METADATA_ATTR = '__lazyllm_tool_runtime_metadata__'
_TOOL_RUNTIME_METADATA_PATCH_ATTR = '__lazyllm_tool_runtime_metadata_patch__'
_FILE_RESOURCE_NAMESPACE = 'file'


class HostFileAccess(str, Enum):
    UNDECLARED = 'UNDECLARED'
    NONE = 'NONE'
    DECLARED = 'DECLARED'
    OPAQUE = 'OPAQUE'


@dataclass(frozen=True)
class HostFileIntent:
    path: str
    operation: str

    def __post_init__(self):
        if not isinstance(self.path, str) or not os.path.isabs(self.path):
            raise ValueError('host file paths must be absolute strings')
        if self.operation not in ('read', 'write', 'delete'):
            raise ValueError('host file operations must be read, write, or delete')


@dataclass(frozen=True)
class HostFileResolution:
    arguments: Dict[str, Any]
    files: Tuple[HostFileIntent, ...] = ()

    def __post_init__(self):
        if not isinstance(self.arguments, dict):
            raise TypeError('resolved arguments must be a dict')
        if not isinstance(self.files, tuple) or not all(isinstance(item, HostFileIntent) for item in self.files):
            raise TypeError('resolved files must be a tuple of HostFileIntent values')

    @property
    def access(self):
        reads = frozenset(_normalize_resource_key(('file', item.path))
                          for item in self.files if item.operation == 'read')
        writes = frozenset(_normalize_resource_key(('file', item.path))
                           for item in self.files if item.operation != 'read')
        return ResolvedToolAccess(read_keys=reads - writes, write_keys=writes)


def _readonly_snapshot(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _readonly_snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_readonly_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _normalize_resource_key(key: Any):
    if isinstance(key, str):
        if not key.strip():
            raise ValueError('resource keys must not contain empty strings')
        return 'exact', key
    if not isinstance(key, tuple) or not key:
        raise TypeError('each resource key must be a non-empty string or tuple')
    if key[0] == _FILE_RESOURCE_NAMESPACE:
        if len(key) != 2:
            raise ValueError('file resource keys must have the form ("file", path)')
        path = os.fspath(key[1])
        if not isinstance(path, str) or not path.strip():
            raise ValueError('file resource key paths must be non-empty strings')
        path = os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(path))))
        return _FILE_RESOURCE_NAMESPACE, Path(path)
    return 'exact', key


def _normalize_resource_keys(value: Any):
    if isinstance(value, (str, tuple)):
        values = (value,)
    elif isinstance(value, (list, set, frozenset)):
        values = value
    else:
        raise TypeError('resource keys must be a string, tuple, list, set, or frozenset')
    if not values:
        raise ValueError('resource keys must not be empty')
    return frozenset(_normalize_resource_key(key) for key in values)


@dataclass(frozen=True)
class ResolvedToolAccess:
    read_keys: frozenset = frozenset()
    write_keys: frozenset = frozenset()
    exclusive: bool = False


@dataclass(frozen=True)
class PreparedToolCall:
    index: int
    tool_call: Dict[str, Any]
    call_id: str
    tool_name: str
    arguments: Any
    validated_arguments: Optional[Dict[str, Any]]
    access: ResolvedToolAccess = field(default_factory=ResolvedToolAccess)
    polling: bool = False
    host_file_access: HostFileAccess = HostFileAccess.UNDECLARED
    host_files: Tuple[HostFileIntent, ...] = ()

    @property
    def ready(self) -> bool:
        return self.validated_arguments is not None


@dataclass(frozen=True)
class PreparedToolBatch:
    _owner: Any = field(repr=False, compare=False)
    _invocations: tuple = field(repr=False)

    def __len__(self):
        return len(self._invocations)

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        prepared = self._invocations[index].prepared
        return PreparedToolCall(
            index=prepared.index,
            tool_call=_readonly_snapshot(prepared.tool_call),
            call_id=prepared.call_id,
            tool_name=prepared.tool_name,
            arguments=_readonly_snapshot(prepared.arguments),
            validated_arguments=_readonly_snapshot(prepared.validated_arguments),
            access=prepared.access,
            polling=prepared.polling,
            host_file_access=prepared.host_file_access,
            host_files=prepared.host_files,
        )


class ToolExecutionDisposition(str, Enum):
    EXECUTED = 'executed'
    PREPARATION_FAILED = 'preparation_failed'
    SKIPPED = 'skipped'


@dataclass(frozen=True)
class ToolExecutionRecord:
    prepared: PreparedToolCall
    result: Any
    disposition: ToolExecutionDisposition = ToolExecutionDisposition.EXECUTED
    reason: str = ''

    @property
    def index(self) -> int:
        return self.prepared.index

    @property
    def call_id(self) -> str:
        return self.prepared.call_id

    @property
    def tool_name(self) -> str:
        return self.prepared.tool_name

    @property
    def arguments(self) -> Any:
        return self.prepared.arguments

    @property
    def validated_arguments(self) -> Optional[Dict[str, Any]]:
        return self.prepared.validated_arguments

    @property
    def access(self) -> ResolvedToolAccess:
        return self.prepared.access

    @property
    def polling(self) -> bool:
        return self.prepared.polling


@dataclass(frozen=True)
class ToolExecutionBatch:
    results: Any
    records: Tuple[ToolExecutionRecord, ...] = ()
    duration_ms: Optional[int] = None

    def stamped_results(self):
        results = self.results
        if self.duration_ms is None:
            return results
        try:
            results.duration_ms = self.duration_ms
            return results
        except (AttributeError, TypeError):
            from lazyllm.common import package
            stamped = package(() if results is None else results)
            stamped.duration_ms = self.duration_ms
            return stamped


@dataclass(frozen=True)
class ToolRuntimeMetadata:
    execute_in_sandbox: bool = True
    input_files_parm: Optional[str] = None
    output_files_parm: Optional[str] = None
    output_files: tuple = ()
    read_keys: Any = None
    write_keys: Any = None
    exclusive: bool = False
    polling: bool = False
    host_file_access: HostFileAccess = HostFileAccess.UNDECLARED
    host_file_resolver: Optional[Callable] = None

    def _validate_host_file_metadata(self):
        object.__setattr__(self, 'host_file_access', HostFileAccess(self.host_file_access))
        if self.host_file_access is HostFileAccess.DECLARED:
            if self.host_file_resolver is None:
                raise ValueError('DECLARED host file access requires a resolver')
            if not callable(self.host_file_resolver):
                raise TypeError('host_file_resolver must be callable')
        elif self.host_file_resolver is not None:
            raise ValueError('only DECLARED host file access accepts a resolver')

    def __post_init__(self):
        self._validate_host_file_metadata()
        if not isinstance(self.execute_in_sandbox, bool):
            raise TypeError('execute_in_sandbox must be a bool')
        for name in ('input_files_parm', 'output_files_parm'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f'{name} must be a string or None')
        if not isinstance(self.output_files, tuple) or not all(isinstance(item, str) for item in self.output_files):
            raise TypeError('output_files must be a tuple of strings')
        if not isinstance(self.exclusive, bool):
            raise TypeError('exclusive must be a bool')
        if not isinstance(self.polling, bool):
            raise TypeError('polling must be a bool')
        if self.exclusive and (self.read_keys is not None or self.write_keys is not None):
            raise ValueError('exclusive cannot be combined with read_keys or write_keys')
        for source in (self.read_keys, self.write_keys):
            if source is not None and not callable(source):
                _normalize_resource_keys(source)

    @staticmethod
    def _resolve_source(source, arguments):
        if source is None:
            return frozenset()
        value = source(arguments) if callable(source) else source
        return _normalize_resource_keys(value)

    def resolve(self, arguments: Dict[str, Any]) -> ResolvedToolAccess:
        if self.exclusive:
            return ResolvedToolAccess(exclusive=True)
        read_keys = self._resolve_source(self.read_keys, arguments)
        write_keys = self._resolve_source(self.write_keys, arguments)
        return ResolvedToolAccess(
            read_keys=read_keys - write_keys,
            write_keys=write_keys,
        )


def _get_tool_runtime_metadata(func: Optional[Callable]) -> Optional[ToolRuntimeMetadata]:
    if func is None:
        return None
    target = getattr(func, '__func__', func)
    try:
        canonical = inspect.unwrap(target)
    except (TypeError, ValueError):
        canonical = target
    return getattr(canonical, _TOOL_RUNTIME_METADATA_ATTR, None) \
        or getattr(target, _TOOL_RUNTIME_METADATA_ATTR, None)


def _set_tool_runtime_metadata(func: Callable, patch: Dict[str, Any]) -> None:
    target = getattr(func, '__func__', func)
    try:
        canonical = inspect.unwrap(target)
    except (TypeError, ValueError):
        canonical = target
    existing_patch = dict(
        getattr(canonical, _TOOL_RUNTIME_METADATA_PATCH_ATTR, None)
        or getattr(target, _TOOL_RUNTIME_METADATA_PATCH_ATTR, {})
        or {}
    )
    for name, value in patch.items():
        if name in existing_patch and existing_patch[name] != value:
            raise ValueError(
                f'conflicting ToolRuntimeMetadata declaration for field {name!r}'
            )
        existing_patch[name] = value
    metadata = ToolRuntimeMetadata(**existing_patch)
    for item in {target, canonical}:
        setattr(item, _TOOL_RUNTIME_METADATA_PATCH_ATTR, existing_patch)
        setattr(item, _TOOL_RUNTIME_METADATA_ATTR, metadata)


def _resource_keys_overlap(left, right) -> bool:
    if left[0] != right[0]:
        return False
    if left[0] != _FILE_RESOURCE_NAMESPACE:
        return left[1] == right[1]
    left_parts, right_parts = left[1].parts, right[1].parts
    common_length = min(len(left_parts), len(right_parts))
    return bool(common_length) and left_parts[:common_length] == right_parts[:common_length]


def _accesses_conflict(current: ResolvedToolAccess, reserved: ResolvedToolAccess) -> bool:
    occupied = reserved.read_keys | reserved.write_keys
    return (
        any(_resource_keys_overlap(write, key) for write in current.write_keys for key in occupied)
        or any(_resource_keys_overlap(read, key) for read in current.read_keys for key in reserved.write_keys)
    )
