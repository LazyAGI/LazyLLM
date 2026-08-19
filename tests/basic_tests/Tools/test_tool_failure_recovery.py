import json
from typing import Dict, Literal

from lazyllm.tools.agent import ToolDomainError, ToolPolicyError
from lazyllm.tools.agent.toolsManager import ToolManager


def typed_search(query: str, limit: int = 10, mode: Literal['semantic', 'keyword'] = 'semantic'):
    '''Search typed data.

    Args:
        query (str): Search query.
        limit (int): Maximum result count.
        mode (str): Search mode.
    '''
    return {'query': query, 'limit': limit, 'mode': mode}


def nested_search(query: str, filters: Dict[str, int]):
    '''Search with nested filters.

    Args:
        query (str): Search query.
        filters (Dict[str, int]): Nested search filters.
    '''
    return {'query': query, 'filters': filters}


def permission_tool(path: str):
    '''Read a protected path.

    Args:
        path (str): Protected path.
    '''
    raise PermissionError('path is outside the allowed scope')


def timeout_tool(query: str):
    '''Call a temporary unavailable service.

    Args:
        query (str): Service query.
    '''
    raise TimeoutError('upstream timed out')


def business_status(resource: str):
    '''Return business data that resembles an error.

    Args:
        resource (str): Resource name.
    '''
    return {
        'success': False,
        'status': 'error',
        'resource': resource,
    }


def typed_domain_failure(resource: str):
    '''Raise a typed domain failure.

    Args:
        resource (str): Resource name.
    '''
    raise ToolDomainError(
        'resource does not exist',
        code='RESOURCE_NOT_FOUND',
        details={'resource_type': 'document', 'resource': resource},
    )


def typed_policy_failure(operation: str):
    '''Reject an operation through runtime policy.

    Args:
        operation (str): Rejected operation.
    '''
    raise ToolPolicyError(
        'operation was blocked by runtime policy',
        code='REPEATED_TOOL_CALL',
        details={'operation': operation},
    )


def flexible_search(query: str, **kwargs):
    '''Search with arbitrary provider options.

    Args:
        query (str): Search query.
    '''
    return {'query': query, **kwargs}


def _call(name, arguments):
    return {
        'id': f'call-{name}',
        'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }


def test_unknown_tool_precedes_argument_validation_and_suggests_visible_tool():
    manager = ToolManager([typed_search])
    call = {'function': {'name': 'typed_seach', 'arguments': '{not-json'}}

    result = manager(call, allowed_tool_names={'typed_search'})[0]

    assert result['error']['category'] == 'UNKNOWN_TOOL'
    assert result['error']['code'] == 'TOOL_NOT_EXPOSED'
    assert result['error']['details'] == {
        'suggested_tool': 'typed_search',
        'edit_distance': 1,
    }
    assert result['error']['recovery_action'] == 'choose_tool'
    assert 'retryable' not in result['error']
    assert 'recovery_attempts_remaining' not in result['error']


def test_invalid_args_return_missing_and_type_violations():
    manager = ToolManager([typed_search])

    result = manager(_call('typed_search', {'limit': 'many'}))[0]

    assert result['error']['category'] == 'INVALID_ARGS'
    assert result['error']['code'] == 'SCHEMA_VALIDATION_FAILED'
    violations = result['error']['details']['violations']
    assert [(item['path'], item['type']) for item in violations] == [
        ('query', 'missing'),
        ('limit', 'int_parsing'),
    ]
    assert all(item['message'] for item in violations)
    assert result['error']['recovery_action'] == 'fix_arguments'
    assert 'query: Field required' in result['error']['message']


def test_invalid_args_return_enum_and_nested_paths():
    typed_manager = ToolManager([typed_search])
    nested_manager = ToolManager([nested_search])

    enum_result = typed_manager(_call('typed_search', {
        'query': 'LazyLLM', 'mode': 'hybrid',
    }))[0]
    nested_result = nested_manager(_call('nested_search', {
        'query': 'LazyLLM', 'filters': {'limit': 'many'},
    }))[0]

    enum_violation = enum_result['error']['details']['violations'][0]
    nested_violation = nested_result['error']['details']['violations'][0]
    assert (enum_violation['path'], enum_violation['type']) == ('mode', 'literal_error')
    assert enum_violation['context']['expected'] == "'semantic' or 'keyword'"
    assert (nested_violation['path'], nested_violation['type']) == ('filters.limit', 'int_parsing')


def test_invalid_json_and_non_object_arguments_are_classified():
    manager = ToolManager([typed_search])
    invalid_json = {
        'function': {'name': 'typed_search', 'arguments': '{not-json'},
    }
    not_object = {
        'function': {'name': 'typed_search', 'arguments': '[]'},
    }

    invalid_result = manager(invalid_json)[0]
    object_result = manager(not_object)[0]

    assert invalid_result['error']['code'] == 'ARGUMENTS_JSON_INVALID'
    assert object_result['error']['code'] == 'ARGUMENTS_NOT_OBJECT'
    assert object_result['error']['details']['violations'][0]['input_type'] == 'array'


def test_fixed_schema_forbids_extra_but_kwargs_accepts_it():
    fixed = ToolManager([typed_search])(
        _call('typed_search', {'query': 'LazyLLM', 'qurey': 'typo'}),
    )[0]
    flexible = ToolManager([flexible_search])(
        _call('flexible_search', {'query': 'LazyLLM', 'provider': 'github'}),
    )[0]

    violation = fixed['error']['details']['violations'][0]
    assert (violation['path'], violation['type']) == ('qurey', 'extra_forbidden')
    assert flexible == {
        'ok': True,
        'value': {'query': 'LazyLLM', 'provider': 'github'},
    }


def test_execution_exceptions_are_classified_without_runtime_retry():
    permission_manager = ToolManager([permission_tool])
    transient_manager = ToolManager([timeout_tool])

    permission_result = permission_manager(_call('permission_tool', {'path': '/root'}))[0]
    transient_result = transient_manager(_call('timeout_tool', {'query': 'status'}))[0]

    assert permission_result['error']['category'] == 'PERMISSION_ERROR'
    assert permission_result['error']['recovery_action'] == 'request_authorization'
    assert transient_result['error']['category'] == 'TRANSIENT_ERROR'
    assert transient_result['error']['recovery_action'] == 'retry_later'


def test_returned_business_dict_is_success_and_typed_failure_is_wrapped():
    business = ToolManager([business_status])(
        _call('business_status', {'resource': 'missing'}),
    )[0]
    failure = ToolManager([typed_domain_failure])(
        _call('typed_domain_failure', {'resource': 'missing'}),
    )[0]['error']

    assert business == {
        'ok': True,
        'value': {'success': False, 'status': 'error', 'resource': 'missing'},
    }
    assert failure == {
        'category': 'DOMAIN_FAILURE',
        'code': 'RESOURCE_NOT_FOUND',
        'tool': 'typed_domain_failure',
        'message': 'resource does not exist',
        'recovery_action': 'change_plan',
        'details': {'resource_type': 'document', 'resource': 'missing'},
    }


def test_typed_policy_failure_is_wrapped():
    failure = ToolManager([typed_policy_failure])(
        _call('typed_policy_failure', {'operation': 'search'}),
    )[0]['error']

    assert failure == {
        'category': 'POLICY_ERROR',
        'code': 'REPEATED_TOOL_CALL',
        'tool': 'typed_policy_failure',
        'message': 'operation was blocked by runtime policy',
        'recovery_action': 'change_plan',
        'details': {'operation': 'search'},
    }
