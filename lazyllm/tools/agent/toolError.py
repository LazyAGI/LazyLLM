from typing import Any, Dict, Iterable, Optional

from lazyllm.common import HandledException


def tool_failure(
    message: str,
    *,
    needs_approval: bool = False,
    missing_env: Optional[Iterable[str]] = None,
    error_type: Optional[str] = None,
    dependency: Optional[str] = None,
    exit_code: Optional[int] = None,
    duration_ms: Optional[int] = None,
    blocked_resource: Optional[str] = None,
    available_commands: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    result = {'ok': False, 'value': str(message)}
    if needs_approval:
        result['needs_approval'] = True
    names = [str(name) for name in (missing_env or []) if name]
    if names:
        result['missing_env'] = names
    if error_type:
        result['error_type'] = str(error_type)
    if dependency:
        result['dependency'] = str(dependency)
    if exit_code is not None:
        result['exit_code'] = int(exit_code)
    if duration_ms is not None:
        result['duration_ms'] = int(duration_ms)
    if blocked_resource:
        result['blocked_resource'] = str(blocked_resource)
    commands = [str(name) for name in (available_commands or []) if name]
    if commands:
        result['available_commands'] = commands
    return result


class ToolExecutionError(HandledException):
    needs_approval = False
    missing_env = ()
    blocker = None
    error_type = None
    dependency = None
    exit_code = None
    duration_ms = None
    blocked_resource = None
    available_commands = ()

    @classmethod
    def not_ready(cls, blocker: Dict[str, Any]) -> 'ToolExecutionError':
        error = cls(str(blocker.get('message') or 'Tool configuration required.'))
        error.blocker = dict(blocker)
        return error

    @classmethod
    def approval_required(cls, message: str) -> 'ToolExecutionError':
        error = cls(message)
        error.needs_approval = True
        return error

    @classmethod
    def with_missing_env(cls, message: str, missing_env: Iterable[str]) -> 'ToolExecutionError':
        error = cls(message)
        error.missing_env = [str(name) for name in missing_env if name]
        return error

    @classmethod
    def with_diagnostics(
        cls,
        message: str,
        *,
        error_type: Optional[str] = None,
        dependency: Optional[str] = None,
        exit_code: Optional[int] = None,
        duration_ms: Optional[int] = None,
        missing_env: Optional[Iterable[str]] = None,
        blocked_resource: Optional[str] = None,
        available_commands: Optional[Iterable[str]] = None,
    ) -> 'ToolExecutionError':
        error = cls(message)
        error.error_type = str(error_type) if error_type else None
        error.dependency = str(dependency) if dependency else None
        error.exit_code = int(exit_code) if exit_code is not None else None
        error.duration_ms = int(duration_ms) if duration_ms is not None else None
        error.missing_env = [str(name) for name in (missing_env or []) if name]
        error.blocked_resource = str(blocked_resource) if blocked_resource else None
        error.available_commands = [str(name) for name in (available_commands or []) if name]
        return error


def exception_failure(tool_name: str, error: Exception) -> Dict[str, Any]:
    # ModuleBase translates ordinary exceptions into ModuleExecutionError and
    # suppresses that wrapper's context for presentation. The wrapped exception
    # is still the semantic tool failure, so unwrap this one framework layer
    # before honoring user-authored ``raise ... from None`` chains below.
    from lazyllm.module.module import ModuleExecutionError
    if isinstance(error, ModuleExecutionError) and error.__context__ is not None:
        error = error.__context__

    causes = []
    seen_ids = set()
    current = error
    while current is not None and id(current) not in seen_ids:
        causes.append(current)
        seen_ids.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None

    typed_error = next((item for item in causes if isinstance(item, ToolExecutionError)), None)
    if typed_error is not None:
        if typed_error.blocker is not None:
            return {'ok': False, 'value': typed_error.blocker, 'needs_configuration': True}
        return tool_failure(
            str(typed_error) or type(typed_error).__name__,
            needs_approval=typed_error.needs_approval,
            missing_env=getattr(typed_error, 'missing_env', ()) or None,
            error_type=getattr(typed_error, 'error_type', None),
            dependency=getattr(typed_error, 'dependency', None),
            exit_code=getattr(typed_error, 'exit_code', None),
            duration_ms=getattr(typed_error, 'duration_ms', None),
            blocked_resource=getattr(typed_error, 'blocked_resource', None),
            available_commands=getattr(typed_error, 'available_commands', ()) or None,
        )

    semantic_error = causes[-1] if causes else error
    error_text = str(semantic_error) or type(semantic_error).__name__
    status_codes = set()
    for item in causes:
        status_code = getattr(item, 'status_code', None)
        response = getattr(item, 'response', None)
        status_code = status_code or getattr(response, 'status_code', None)
        if isinstance(status_code, int):
            status_codes.add(status_code)

    status = f' (HTTP {min(status_codes)})' if status_codes else ''
    return tool_failure(f'{tool_name} failed{status}: {error_text}')
