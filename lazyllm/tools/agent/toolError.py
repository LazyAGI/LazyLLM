from typing import Any, Dict

from lazyllm.common import HandledException


def tool_failure(message: str, *, needs_approval: bool = False) -> Dict[str, Any]:
    result = {'ok': False, 'value': str(message)}
    if needs_approval:
        result['needs_approval'] = True
    return result


class ToolExecutionError(HandledException):
    needs_approval = False

    @classmethod
    def approval_required(cls, message: str) -> 'ToolExecutionError':
        error = cls(message)
        error.needs_approval = True
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
        return tool_failure(
            str(typed_error) or type(typed_error).__name__,
            needs_approval=typed_error.needs_approval,
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
