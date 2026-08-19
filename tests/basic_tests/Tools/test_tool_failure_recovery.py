import json
from typing import Dict, Literal

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


def hidden_read(path: str):
    '''Read hidden data.

    Args:
        path (str): Hidden path.
    '''
    return path


def status_error_tool(resource: str):
    '''Return a legacy domain failure.

    Args:
        resource (str): Resource name.
    '''
    return {
        'success': True,
        'tool': 'status_error_tool',
        'result': {
            'status': 'error',
            'message': 'resource does not exist',
            'resource_type': 'document',
        },
    }


def rejected_tool(resource: str):
    '''Reject a missing resource.

    Args:
        resource (str): Resource name.
    '''
    return {
        'success': False,
        'tool': 'rejected_tool',
        'error': {'reason': 'resource does not exist'},
    }


def reported_permission_tool(resource: str):
    '''Return a reported permission failure.

    Args:
        resource (str): Resource name.
    '''
    return {
        'success': False,
        'tool': 'reported_permission_tool',
        'error': {
            'reason': 'authorization is required',
            'type': 'PermissionError',
            'required_capability': 'document.read',
            'authorization_required': True,
        },
    }


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
    assert result['error']['recovery_attempts_remaining'] == 1


def test_allowed_tool_snapshot_is_fixed_for_one_parallel_call():
    manager = ToolManager([{
        'name': 'private',
        'desc': 'Private tools.',
        'lazy': True,
        'prefix': False,
        'tools': [hidden_read],
    }])

    allowed = {item['function']['name'] for item in manager.tools_description}
    results = manager([
        _call('get_private_methods', {}),
        _call('hidden_read', {'path': '/tmp/secret'}),
    ], allowed_tool_names=allowed)

    assert allowed == {'get_private_methods'}
    assert results[0]['ok'] is True
    assert results[1]['error']['category'] == 'UNKNOWN_TOOL'
    assert results[1]['error']['details']['suggested_tool'] is None


def test_invalid_args_return_missing_and_type_violations():
    manager = ToolManager([typed_search])

    result = manager(_call('typed_search', {'limit': 'many'}))[0]

    assert result['error']['category'] == 'INVALID_ARGS'
    assert result['error']['code'] == 'SCHEMA_VALIDATION_FAILED'
    assert result['error']['details']['violations'] == [
        {
            'path': 'query',
            'type': 'missing',
            'actual': 'missing',
            'expected': 'string',
        },
        {
            'path': 'limit',
            'type': 'type_error',
            'actual': 'string',
            'expected': 'integer',
        },
    ]
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

    assert enum_result['error']['details']['violations'] == [{
        'path': 'mode',
        'type': 'enum_error',
        'actual': 'string',
        'expected': {'enum': ['semantic', 'keyword']},
    }]
    assert nested_result['error']['details']['violations'] == [{
        'path': 'filters.limit',
        'type': 'type_error',
        'actual': 'string',
        'expected': 'integer',
    }]


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
    assert object_result['error']['details']['violations'][0]['actual'] == 'array'


def test_execution_exceptions_are_classified_without_runtime_retry():
    permission_manager = ToolManager([permission_tool])
    transient_manager = ToolManager([timeout_tool])

    permission_result = permission_manager(_call('permission_tool', {'path': '/root'}))[0]
    transient_result = transient_manager(_call('timeout_tool', {'query': 'status'}))[0]

    assert permission_result['error']['category'] == 'PERMISSION_ERROR'
    assert permission_result['error']['retryable'] is False
    assert transient_result['error']['category'] == 'TRANSIENT_ERROR'
    assert transient_result['error']['retryable'] is False
    assert transient_result['error']['recovery_attempts_remaining'] == 0


def test_legacy_reported_failures_are_normalized():
    reported = ToolManager([rejected_tool])(
        _call('rejected_tool', {'resource': 'missing'}),
    )[0]['error']
    nested = ToolManager([status_error_tool])(
        _call('status_error_tool', {'resource': 'missing'}),
    )[0]['error']
    permission = ToolManager([reported_permission_tool])(
        _call('reported_permission_tool', {'resource': 'private'}),
    )[0]['error']

    assert (reported['category'], reported['message']) == (
        'DOMAIN_FAILURE', 'resource does not exist',
    )
    assert nested['category'] == 'DOMAIN_FAILURE'
    assert nested['details']['resource_type'] == 'document'
    assert permission['category'] == 'PERMISSION_ERROR'
    assert permission['retryable'] is False
    assert permission['details'] == {
        'error_type': 'PermissionError',
        'required_capability': 'document.read',
        'authorization_required': True,
    }
