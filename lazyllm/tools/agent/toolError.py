from typing import Any, Dict, Optional


_CATEGORY_RECOVERY_ACTION = {
    'UNKNOWN_TOOL': 'choose_tool',
    'INVALID_ARGS': 'fix_arguments',
    'TRANSIENT_ERROR': 'retry_later',
    'PERMISSION_ERROR': 'request_authorization',
    'DOMAIN_FAILURE': 'change_plan',
    'POLICY_ERROR': 'change_plan',
}
_RECOVERY_ACTIONS = frozenset(_CATEGORY_RECOVERY_ACTION.values())

_TRANSIENT_ERROR_NAMES = {
    'ConnectTimeout', 'ConnectTimeoutError', 'ConnectionError', 'ConnectionResetError',
    'ReadTimeout', 'ReadTimeoutError', 'Timeout', 'TimeoutError',
}


def tool_failure(category: str, code: str, tool_name: str, message: str, *,
                 recovery_action: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if category not in _CATEGORY_RECOVERY_ACTION:
        raise ValueError(f'unsupported tool error category: {category}')
    action = recovery_action or _CATEGORY_RECOVERY_ACTION[category]
    if action not in _RECOVERY_ACTIONS:
        raise ValueError(f'unsupported tool recovery action: {action}')
    error = {
        'category': category,
        'code': code,
        'tool': tool_name,
        'message': message,
        'recovery_action': action,
        'details': dict(details or {}),
    }
    return {'ok': False, 'value': None, 'error': error, 'msg': message}


class ToolExecutionError(Exception):
    category = 'DOMAIN_FAILURE'
    default_code = 'TOOL_EXECUTION_FAILED'
    recovery_action = 'change_plan'

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 recovery_action: Optional[str] = None):
        action = recovery_action or type(self).recovery_action
        if action not in _RECOVERY_ACTIONS:
            raise ValueError(f'unsupported tool recovery action: {action}')
        super().__init__(message, code, details, recovery_action)
        self.message = message
        self.code = code or self.default_code
        self.details = dict(details or {})
        self.recovery_action = action

    def __str__(self) -> str:
        return self.message


class ToolInvalidArgumentsError(ToolExecutionError):
    category = 'INVALID_ARGS'
    default_code = 'INVALID_TOOL_ARGUMENTS'
    recovery_action = 'fix_arguments'


class ToolTransientError(ToolExecutionError):
    category = 'TRANSIENT_ERROR'
    default_code = 'TEMPORARY_TOOL_FAILURE'
    recovery_action = 'retry_later'


class ToolPermissionError(ToolExecutionError):
    category = 'PERMISSION_ERROR'
    default_code = 'PERMISSION_DENIED'
    recovery_action = 'request_authorization'


class ToolDomainError(ToolExecutionError):
    default_code = 'TOOL_DOMAIN_FAILURE'


class ToolPolicyError(ToolExecutionError):
    category = 'POLICY_ERROR'
    default_code = 'TOOL_POLICY_VIOLATION'


def exception_failure(tool_name: str, error: Exception) -> Dict[str, Any]:
    causes = []
    current = error
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__

    typed_error = next((item for item in causes if isinstance(item, ToolExecutionError)), None)
    if typed_error is not None:
        return tool_failure(
            typed_error.category,
            typed_error.code,
            tool_name,
            str(typed_error),
            recovery_action=typed_error.recovery_action,
            details=typed_error.details,
        )

    error_names = {type(item).__name__ for item in causes}
    error_text = str(causes[-1]) if causes else str(error)
    status_codes = set()
    for item in causes:
        status_code = getattr(item, 'status_code', None)
        response = getattr(item, 'response', None)
        status_code = status_code or getattr(response, 'status_code', None)
        if isinstance(status_code, int):
            status_codes.add(status_code)

    if any(isinstance(item, PermissionError) for item in causes) or any(
        'Permission' in name or 'Forbidden' in name for name in error_names
    ) or bool(status_codes & {401, 403}):
        details = {'status_code': min(status_codes)} if status_codes else {}
        if 401 in status_codes:
            details['authorization_required'] = True
        return tool_failure(
            'PERMISSION_ERROR', 'PERMISSION_DENIED', tool_name,
            f'{tool_name} is not permitted: {error_text}', details=details,
        )

    if any(isinstance(item, (TimeoutError, ConnectionError)) for item in causes) or \
            bool(error_names & _TRANSIENT_ERROR_NAMES) or bool(status_codes & {408, 429, 502, 503, 504}):
        details = {'status_code': min(status_codes)} if status_codes else {}
        return tool_failure(
            'TRANSIENT_ERROR', 'TEMPORARY_TOOL_FAILURE', tool_name,
            f'{tool_name} failed temporarily: {error_text}', details=details,
        )

    return tool_failure(
        'DOMAIN_FAILURE', 'TOOL_EXECUTION_FAILED', tool_name,
        f'{tool_name} failed: {error_text}',
    )
