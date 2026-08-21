import json
from types import SimpleNamespace
from typing import Dict, Literal

import pytest

from lazyllm.tools.agent import ToolExecutionError
from lazyllm.tools.agent.toolsManager import ToolManager
from lazyllm.tools.git import GitLab, LocalGit
from lazyllm.tools.git.review.poster import _submit_review


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
    raise ToolExecutionError(f'Document resource {resource!r} does not exist.')


def typed_policy_failure(operation: str):
    '''Reject an operation through runtime policy.

    Args:
        operation (str): Rejected operation.
    '''
    raise ToolExecutionError(f'Operation {operation!r} was blocked by runtime policy.')


def flexible_search(query: str, **kwargs):
    '''Search with arbitrary provider options.

    Args:
        query (str): Search query.
    '''
    return {'query': query, **kwargs}


def translated_permission_failure(resource: str):
    '''Translate a domain failure into a permission failure.

    Args:
        resource (str): Protected resource.
    '''
    try:
        raise ToolExecutionError('Resource does not exist.')
    except ToolExecutionError:
        raise PermissionError(f'access denied: {resource}') from None


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

    assert 'Did you mean [typed_search]?' in result['message']
    assert set(result) == {'ok', 'message'}


@pytest.mark.parametrize('tool_call', [
    None,
    {},
    {'function': None},
    {'function': []},
    {'function': {'name': None, 'arguments': '{}'}},
    {'function': {'name': 123, 'arguments': '{}'}},
    {'function': {'name': [], 'arguments': '{}'}},
    {'function': {'name': {}, 'arguments': '{}'}},
    {'function': {'name': '   ', 'arguments': '{}'}},
])
def test_malformed_tool_call_returns_structured_format_failure(tool_call):
    result = ToolManager([typed_search])(tool_call)[0]

    assert result['ok'] is False
    assert 'tool call' in result['message'].lower()
    assert set(result) == {'ok', 'message'}


def test_tool_manager_normalizes_missing_and_dictionary_arguments():
    manager = ToolManager([typed_search])
    missing_arguments = {'function': {'name': 'typed_search'}}
    dictionary_arguments = {
        'function': {'name': 'typed_search', 'arguments': {'query': 'LazyLLM'}},
    }

    missing_result = manager(missing_arguments)[0]
    dictionary_result = manager(dictionary_arguments)[0]

    assert missing_arguments['function']['arguments'] == '{}'
    assert 'query: Field required' in missing_result['message']
    assert dictionary_arguments['function']['arguments'] == '{"query": "LazyLLM"}'
    assert dictionary_result == {
        'ok': True,
        'value': {'query': 'LazyLLM', 'limit': 10, 'mode': 'semantic'},
    }


def test_invalid_args_return_missing_and_type_violations():
    manager = ToolManager([typed_search])

    result = manager(_call('typed_search', {'limit': 'many'}))[0]

    assert 'query: Field required' in result['message']
    assert 'limit:' in result['message']
    assert set(result) == {'ok', 'message'}


def test_invalid_args_return_enum_and_nested_paths():
    typed_manager = ToolManager([typed_search])
    nested_manager = ToolManager([nested_search])

    enum_result = typed_manager(_call('typed_search', {
        'query': 'LazyLLM', 'mode': 'hybrid',
    }))[0]
    nested_result = nested_manager(_call('nested_search', {
        'query': 'LazyLLM', 'filters': {'limit': 'many'},
    }))[0]

    assert 'mode:' in enum_result['message']
    assert "'semantic' or 'keyword'" in enum_result['message']
    assert 'filters.limit:' in nested_result['message']


def test_invalid_json_and_non_object_arguments_return_clear_messages():
    manager = ToolManager([typed_search])
    invalid_json = {
        'function': {'name': 'typed_search', 'arguments': '{not-json'},
    }
    not_object = {
        'function': {'name': 'typed_search', 'arguments': '[]'},
    }

    invalid_result = manager(invalid_json)[0]
    object_result = manager(not_object)[0]

    assert 'valid JSON' in invalid_result['message']
    assert 'JSON object, got list' in object_result['message']


def test_repairable_json_is_parsed_before_schema_validation():
    manager = ToolManager([typed_search])

    repaired = manager({
        'function': {
            'name': 'typed_search',
            'arguments': '{"query": "LazyLLM", "limit": 5,}',
        },
    })[0]

    assert repaired == {
        'ok': True,
        'value': {'query': 'LazyLLM', 'limit': 5, 'mode': 'semantic'},
    }


def test_fixed_schema_forbids_extra_but_kwargs_accepts_it():
    fixed = ToolManager([typed_search])(
        _call('typed_search', {'query': 'LazyLLM', 'qurey': 'typo'}),
    )[0]
    flexible = ToolManager([flexible_search])(
        _call('flexible_search', {'query': 'LazyLLM', 'provider': 'github'}),
    )[0]

    assert 'qurey:' in fixed['message']
    assert flexible == {
        'ok': True,
        'value': {'query': 'LazyLLM', 'provider': 'github'},
    }


def test_execution_exceptions_return_clear_messages_without_runtime_retry():
    permission_manager = ToolManager([permission_tool])
    transient_manager = ToolManager([timeout_tool])

    permission_result = permission_manager(_call('permission_tool', {'path': '/root'}))[0]
    transient_result = transient_manager(_call('timeout_tool', {'query': 'status'}))[0]

    assert 'outside the allowed scope' in permission_result['message']
    assert 'upstream timed out' in transient_result['message']


def test_exception_translation_from_none_hides_suppressed_context():
    manager = ToolManager([translated_permission_failure])

    result = manager(_call('translated_permission_failure', {'resource': 'secret'}))[0]

    assert 'access denied' in result['message']


def test_returned_business_dict_is_success_and_typed_failure_is_wrapped():
    business = ToolManager([business_status])(
        _call('business_status', {'resource': 'missing'}),
    )[0]
    failure = ToolManager([typed_domain_failure])(
        _call('typed_domain_failure', {'resource': 'missing'}),
    )[0]

    assert business == {
        'ok': True,
        'value': {'success': False, 'status': 'error', 'resource': 'missing'},
    }
    assert failure == {'ok': False, 'message': "Document resource 'missing' does not exist."}


def test_typed_policy_failure_is_wrapped():
    failure = ToolManager([typed_policy_failure])(
        _call('typed_policy_failure', {'operation': 'search'}),
    )[0]

    assert failure == {'ok': False, 'message': "Operation 'search' was blocked by runtime policy."}


def test_git_sdk_and_tool_manager_share_typed_failure_contract():
    backend = LocalGit()

    def add_issue_comment(number: int, body: str):
        '''Add an issue comment through the Git SDK.

        Args:
            number (int): Issue number.
            body (str): Comment body.
        '''
        return backend.add_issue_comment(number, body)

    manager = ToolManager([add_issue_comment])
    call = _call('add_issue_comment', {'number': 1, 'body': 'comment'})

    with pytest.raises(ToolExecutionError) as exc_info:
        backend.add_issue_comment(1, 'comment')
    result = manager(call)[0]

    assert 'add_issue_comment' in str(exc_info.value)
    assert 'add_issue_comment' in result['message']
    assert set(result) == {'ok', 'message'}


def test_git_success_remains_normal_business_data():
    result = LocalGit().list_issue_comments(1)

    assert result == {
        'success': True,
        'comments': [],
    }


class _GitFailureStub(LocalGit):
    def permission_failure(self):
        response = SimpleNamespace(status_code=403, text='provider-specific rejection', reason='')
        return self._http_failure(response)

    def transient_failure(self):
        response = SimpleNamespace(status_code=503, text='provider-specific outage', reason='')
        return self._http_failure(response)

    def rate_limit_failure(self):
        response = SimpleNamespace(
            status_code=403,
            text='API rate limit exceeded',
            reason='',
            headers={'X-RateLimit-Remaining': '0'},
        )
        return self._http_failure(response)

    def domain_failure(self):
        return {'success': False, 'message': 'reference was not found', 'status_code': 404}

    def runtime_value_error(self):
        raise ValueError('invalid provider response')


@pytest.mark.parametrize('method_name', [
    'permission_failure',
    'transient_failure',
    'rate_limit_failure',
    'domain_failure',
    'runtime_value_error',
])
def test_git_sdk_normalizes_failures_at_direct_call_boundary(method_name):
    with pytest.raises(ToolExecutionError) as exc_info:
        getattr(_GitFailureStub(), method_name)()

    assert method_name in str(exc_info.value)


def test_git_base_public_apis_and_explicit_validation_use_typed_failures():
    backend = LocalGit()

    for method_name in (
        'check_review_resolution', 'stash_review_comment', 'submit_review_with_comments'
    ):
        assert getattr(getattr(backend, method_name), '__git_failure_boundary__', False)

    with pytest.raises(ToolExecutionError) as stash_error:
        backend.stash_review_comment(1, 'comment', 'file.py')
    with pytest.raises(ToolExecutionError) as remote_error:
        backend.push_branch('feature', remote_name='ext::helper')

    assert 'repo is not set' in str(stash_error.value)
    assert 'ext::helper' in str(remote_error.value)


@pytest.mark.parametrize('status_code', [403, 503])
def test_git_provider_http_status_is_preserved_in_message(monkeypatch, status_code):
    backend = GitLab(token='token', repo='owner/repo')
    response = SimpleNamespace(
        status_code=status_code,
        text='opaque provider response',
        reason='provider failure',
    )
    monkeypatch.setattr(backend, '_req', lambda *args, **kwargs: response)

    with pytest.raises(ToolExecutionError) as exc_info:
        backend.create_pull_request('feature', 'main', 'title')

    assert f'HTTP {status_code}' in str(exc_info.value)
    assert 'create_pull_request' in str(exc_info.value)


@pytest.mark.parametrize('status_code', [403, 503])
def test_git_provider_helper_preserves_http_status_message(monkeypatch, status_code):
    backend = GitLab(token='token')
    response = SimpleNamespace(
        status_code=status_code,
        text='opaque provider response',
        reason='provider failure',
    )
    monkeypatch.setattr(backend._session, 'get', lambda *args, **kwargs: response)

    with pytest.raises(ToolExecutionError) as exc_info:
        backend.list_user_starred_repos()

    assert f'HTTP {status_code}' in str(exc_info.value)
    assert 'list_user_starred_repos' in str(exc_info.value)


@pytest.mark.parametrize('submit_kwargs', [
    {'body': 'review body'},
    {'comments': [{'body': 'fallback comment'}]},
])
@pytest.mark.parametrize('status_code', [403, 503])
def test_gitlab_submit_review_checks_note_write_failures(
        monkeypatch, submit_kwargs, status_code):
    backend = GitLab(token='token', repo='owner/repo')
    response = SimpleNamespace(
        status_code=status_code,
        text='opaque provider response',
        reason='provider failure',
    )
    monkeypatch.setattr(backend, '_req', lambda *args, **kwargs: response)

    with pytest.raises(ToolExecutionError) as exc_info:
        backend.submit_review(1, 'COMMENT', **submit_kwargs)

    assert f'HTTP {status_code}' in str(exc_info.value)
    assert 'submit_review' in str(exc_info.value)


def test_batch_review_comments_preserves_typed_failure_and_partial_progress(monkeypatch):
    backend = GitLab(token='token', repo='owner/repo')
    responses = iter([
        SimpleNamespace(status_code=201, text='', reason='', json=lambda: {'id': 1}),
        SimpleNamespace(status_code=403, text='opaque provider response', reason='provider failure'),
    ])
    monkeypatch.setattr(backend, '_req', lambda *args, **kwargs: next(responses))
    backend.stash_review_comment(1, 'first', 'first.py', 1)
    backend.stash_review_comment(1, 'second', 'second.py', 2)

    with pytest.raises(ToolExecutionError) as exc_info:
        backend.batch_commit_review_comments()

    error = exc_info.value
    assert '1 created and 1 failed' in str(error)
    assert 'HTTP 403' in str(error)
    assert 'create_review_comment' in str(error)
    assert backend._stashed_comments() == []


def test_review_submit_reports_failure_without_classification():
    class Backend:
        def __init__(self):
            self.calls = 0

        def submit_review(self, **_kwargs):
            self.calls += 1
            raise ToolExecutionError('GitHub API rate limit exceeded.')

    backend = Backend()

    assert _submit_review(backend, 1, 'sha', [], 'body') is False
    assert backend.calls == 1
