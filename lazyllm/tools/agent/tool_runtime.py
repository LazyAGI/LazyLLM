import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple


_TOOL_RUNTIME_METADATA_ATTR = '__lazyllm_tool_runtime_metadata__'
_TOOL_RUNTIME_METADATA_PATCH_ATTR = '__lazyllm_tool_runtime_metadata_patch__'
_FILE_RESOURCE_NAMESPACE = 'file'


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
    polling: bool = False


@dataclass(frozen=True)
class PreparedToolCall:
    tool_call: Dict[str, Any]
    call_id: str
    tool_name: str
    arguments: Any
    validated_arguments: Optional[Dict[str, Any]]
    access: ResolvedToolAccess = field(default_factory=ResolvedToolAccess)
    failure: Any = None

    @property
    def ready(self) -> bool:
        return self.failure is None and self.validated_arguments is not None

    @property
    def preparation_status(self) -> str:
        return 'ready' if self.ready else 'invalid'


@dataclass(frozen=True)
class ToolExecutionRecord:
    prepared: PreparedToolCall
    result: Any
    executed: bool = True

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


@dataclass(frozen=True)
class ToolExecutionBatch:
    results: Any
    records: Tuple[ToolExecutionRecord, ...] = ()


@dataclass(frozen=True)
class RuntimeContext:
    content: str

    def __post_init__(self):
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError('runtime context content must be a non-empty string')


@dataclass(frozen=True)
class RuntimeDelta:
    model_context: Tuple[RuntimeContext, ...] = ()


class AgentRuntimeExtension(Protocol):
    def begin_run(self, context: Dict[str, Any]) -> None: ...

    def after_tool_batch(self, records: Sequence[ToolExecutionRecord]) -> RuntimeDelta: ...

    def end_run(self, reason: str) -> None: ...


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

    def __post_init__(self):
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
            return ResolvedToolAccess(exclusive=True, polling=self.polling)
        read_keys = self._resolve_source(self.read_keys, arguments)
        write_keys = self._resolve_source(self.write_keys, arguments)
        return ResolvedToolAccess(
            read_keys=read_keys - write_keys,
            write_keys=write_keys,
            polling=self.polling,
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
