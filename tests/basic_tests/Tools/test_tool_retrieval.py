import pytest
import lazyllm
from lazyllm.tools.agent.toolsManager import ToolManager
from lazyllm.tools.agent.toolError import ToolExecutionError


def find_mail(query: str) -> str:
    '''Search email messages by subject.

    Args:
        query (str): Email search keywords.
    '''
    return query


def read_mail(message_id: str) -> str:
    '''Read an email message body.

    Args:
        message_id (str): Email message identifier.
    '''
    return message_id


def manager(**options):
    tool = ToolManager([{'name': 'Mail', 'desc': 'Search and read email messages.',
                         'tools': [find_mail, read_mail]}])
    retrieval = tool.enable_tool_retrieval(
        required=options.pop('required', []), groups={'Mail'},
        estimate_tokens=lambda schemas: len(schemas) * 10,
        threshold_tokens=options.pop('threshold_tokens', 30), **options)
    return tool, retrieval


def test_search_group_partial_load_and_next_round_snapshot():
    tool, retrieval = manager()
    visible = {d['function']['name'] for d in tool.tools_description}
    assert visible == {'search_tools', 'load_tools'}
    assert retrieval.search('email', 5, 'short')[0]['name'] == 'Mail'
    assert visible == {d['function']['name'] for d in tool.tools_description}
    retrieval.load(['find_mail'], [])
    assert retrieval.search('email', 5, 'long')[0]['name'] == 'Mail'
    # Even though the callable is registered and now loaded, the old model snapshot rejects it.
    denied = tool([{'id': '1', 'function': {'name': 'find_mail', 'arguments': '{"query":"hello"}'}}],
                  allowed_tool_names=visible)
    assert denied[0]['ok'] is False
    retrieval.load(['Mail'], [])
    assert retrieval.search('email', 5, 'short') == []
    retrieval.load([], ['find_mail'])
    tool.sync_active_groups(history=[{'tool_calls': [{'function': {'name': 'get_Mail_methods'}}]}])
    assert 'find_mail' not in {d['function']['name'] for d in tool.tools_description}
    assert 'get_Mail_methods' not in tool.tools_info


def test_budget_atomicity_protection_and_required_preload():
    tool, retrieval = manager(threshold_tokens=20)
    assert retrieval.load(['Mail'], [])['over_threshold'] is True
    assert retrieval.load(['find_mail'], [])['status'] == 'ok'
    with pytest.raises(ToolExecutionError):
        retrieval.load(['unknown'], ['Mail'])
    assert len(tool.tools_description) == 4
    with pytest.raises(ToolExecutionError):
        retrieval.load([], ['load_tools'])
    retrieval.load([], ['read_mail'])
    with pytest.raises(ToolExecutionError):
        retrieval.load(['read_mail'], [])
    assert retrieval.load(['read_mail'], ['find_mail'])['status'] == 'ok'
    _, mandatory = manager(required=['find_mail', 'read_mail'], threshold_tokens=1)
    assert len(mandatory.descriptions()) == 4
    with pytest.raises(ToolExecutionError):
        mandatory.load([], ['Mail'])


def test_session_isolation_and_provider_selection():
    tool = ToolManager([{'name': 'Provider', 'desc': 'Email search provider.', 'pick_first_valid': True,
                         'tools': [(find_mail, lambda: False), read_mail]}])
    assert list(tool.atomic_tool_catalog()) == ['read_mail']
    _, retrieval = manager()
    old_sid = lazyllm.locals._sid
    try:
        lazyllm.locals._init_sid('retrieval-a')
        retrieval.load(['find_mail'], [])
        lazyllm.locals._init_sid('retrieval-b')
        assert len(retrieval.descriptions()) == 2
    finally:
        lazyllm.locals._init_sid(old_sid)


def test_function_call_load_then_call_uses_new_schema_next_round():
    from lazyllm.tools.agent.functionCall import FunctionCall

    class ScriptedModel:
        _module_id = 'retrieval-scripted-model'

        def share(self, **kwargs):
            return self

        def used_by(self, module_id):
            return self

    tool, retrieval = manager()
    lazyllm.locals['_lazyllm_agent'] = {'workspace': {}}
    fc = FunctionCall(ScriptedModel(), _tool_manager=tool)
    fc._get_current_tools(refresh=True)
    fc._post_action({'content': '', 'tool_calls': [
        {'id': 'load', 'function': {'name': 'load_tools', 'arguments': '{"tool_names":["find_mail"]}'}},
        {'id': 'early', 'function': {'name': 'find_mail', 'arguments': '{"query":"hello"}'}},
    ]})
    results = lazyllm.locals['_lazyllm_agent']['workspace']['tool_call_trace']
    assert results[0]['tool_call_result']['ok'] is True
    assert results[1]['tool_call_result']['ok'] is False
    assert 'find_mail' not in fc._get_visible_tool_names()
    fc._get_current_tools(refresh=True)
    assert 'find_mail' in fc._get_visible_tool_names()
    fc._post_action({'content': '', 'tool_calls': [
        {'id': 'next', 'function': {'name': 'find_mail', 'arguments': '{"query":"hello"}'}},
    ]})
    assert lazyllm.locals['_lazyllm_agent']['workspace']['tool_call_trace'][0]['tool_call_result']['ok'] is True


def test_group_deduplication_happens_before_top_k():
    groups = {f'Mail{index}' for index in range(7)}
    tool = ToolManager([{'name': name, 'desc': 'Search email messages.', 'prefix': True,
                         'tools': [find_mail, read_mail]} for name in sorted(groups)])
    retrieval = tool.enable_tool_retrieval(required=[], groups=groups,
                                           estimate_tokens=lambda schemas: len(schemas), threshold_tokens=100)
    found = retrieval.search('email', 5, 'short')
    assert len(found) == 5
    assert len({item['name'] for item in found}) == 5
    assert all(item['type'] == 'group' for item in found)
    assert found == retrieval.search('email', 5, 'short')
    assert retrieval.search('unmatchablezz', 5, 'short') == []


def test_group_document_and_member_explanations():
    tool = ToolManager([{'name': 'Mail', 'desc': 'Correspondence orchestration.',
                         'tools': [find_mail, read_mail]}])
    retrieval = tool.enable_tool_retrieval(
        required=[], groups={'Mail'}, group_descriptions={'Mail': 'Manage your inbox.'},
        estimate_tokens=len, threshold_tokens=100)
    result = retrieval.search('correspondence', 5, 'long')
    assert result == [{'name': 'Mail', 'type': 'group', 'description': 'Manage your inbox.',
                       'matched_members': []}]
    result = retrieval.search('subject', 5, 'short')[0]
    assert result['name'] == 'Mail'
    assert result['matched_members'] == [{'name': 'find_mail',
                                         'description': 'Search email messages by subject.'}]
    retrieval.load(['find_mail'], [])
    assert retrieval.search('subject', 5, 'short') == []
    assert retrieval.search('correspondence', 5, 'short')[0]['matched_members'] == []
    retrieval.load([], ['find_mail'])
    assert retrieval.search('subject', 5, 'short')[0]['matched_members'][0]['name'] == 'find_mail'
    retrieval.load(['Mail'], [])
    assert retrieval.search('correspondence', 5, 'short') == []


def test_nested_group_description_and_provider_refresh():
    available = [True]
    tool = ToolManager([{'name': 'Business', 'desc': 'Correspondence orchestration.',
                         'pick_first_valid': True, 'tools': [
                             ({'name': 'First', 'desc': 'First supplier.', 'tools': [find_mail]},
                              lambda: available[0]),
                             {'name': 'Second', 'desc': 'Second supplier.', 'tools': [read_mail]},
                         ]}])
    retrieval = tool.enable_tool_retrieval(required=[], groups={'Business'},
                                           estimate_tokens=len, threshold_tokens=100)
    assert retrieval.search('correspondence', 5, 'long')[0]['description'] == 'Correspondence orchestration.'
    assert retrieval.search('subject', 5, 'short')[0]['matched_members'][0]['name'] == 'find_mail'
    available[0] = False
    assert retrieval.search('subject', 5, 'short') == []
    assert retrieval.search('body', 5, 'short')[0]['matched_members'][0]['name'] == 'read_mail'
    retrieval.load(['Business'], [])
    assert {d['function']['name'] for d in tool.tools_description} == {'search_tools', 'load_tools', 'read_mail'}


def test_member_summary_cap_and_partial_load():
    def member(name):
        def search(query: str) -> str:
            '''Search messages.

            Args:
                query (str): Keywords to search.
            '''
            return query
        search.__name__ = name
        return search
    tool = ToolManager([{'name': name, 'desc': 'Correspondence service.',
                         'tools': [member(f'{name}_{i}') for i in range(4)]} for name in ('A', 'B')])
    retrieval = tool.enable_tool_retrieval(required=[], groups={'A', 'B'}, estimate_tokens=len, threshold_tokens=100)
    before = retrieval.search('messages', 5, 'short')
    assert [r['name'] for r in before] == ['A', 'B']
    assert [r['name'] for r in before[0]['matched_members']] == ['A_0', 'A_1', 'A_2']
    retrieval.load(['A_0'], [])
    after = retrieval.search('messages', 5, 'short')
    assert {r['name'] for r in after} == {'A', 'B'}
    remaining = next(r for r in after if r['name'] == 'A')
    assert [r['name'] for r in remaining['matched_members']] == ['A_1', 'A_2', 'A_3']


def test_disabled_retrieval_gateway_still_activates():
    tool = ToolManager([{'name': 'Mail', 'desc': 'Call the gateway before reading mail.',
                         'lazy': True, 'tools': [find_mail, read_mail]}])
    lazyllm.locals['_lazyllm_agent'] = {'workspace': {}}
    assert [d['function']['name'] for d in tool.tools_description] == ['get_Mail_methods']
    result = tool([{'id': 'gateway', 'function': {'name': 'get_Mail_methods', 'arguments': '{}'}}])
    assert result[0]['ok'] is True
    assert {d['function']['name'] for d in tool.tools_description} == {'find_mail', 'read_mail'}


def test_dynamic_groups_only_expand_available_atomic_members():
    tool = ToolManager([find_mail, read_mail])
    retrieval = tool.enable_tool_retrieval(
        required=[], groups=set(), group_members={'mcp:server-id': ['find_mail', 'read_mail', 'revoked']},
        group_descriptions={'mcp:server-id': 'Mailbox service.'},
        estimate_tokens=len, threshold_tokens=100)
    assert retrieval.search('mailbox', 5, 'long')[0]['name'] == 'mcp:server-id'
    retrieval.load(['find_mail'], [])
    assert retrieval.search('subject', 5, 'short') == []
    assert retrieval.load(['mcp:server-id'], [])['loaded'] == ['find_mail', 'read_mail']
    assert retrieval.search('mailbox', 5, 'short') == []
    assert set(tool.tools_info) == {'find_mail', 'read_mail', 'search_tools', 'load_tools'}
    with pytest.raises(ToolExecutionError):
        retrieval.load(['revoked'], [])


def test_load_validation_preserves_local_state_and_skill_dependencies():
    def validate(definitions):
        if any(d['function']['name'] == 'read_mail' for d in definitions):
            raise ToolExecutionError('fixed context exceeds limit')

    tool, retrieval = manager(validate_load=validate)
    retrieval.initialize()
    retrieval.load(['find_mail'], [])
    before = retrieval.descriptions()
    with pytest.raises(ToolExecutionError, match='fixed context'):
        retrieval.load(['read_mail'], ['find_mail'])
    assert retrieval.descriptions() == before
    with pytest.raises(ToolExecutionError, match='fixed context'):
        retrieval.load_skill('mail', ['read_mail'])
    assert retrieval._read()['skills'] == {}
    assert retrieval.descriptions() == before
    _, mandatory = manager(required=['read_mail'], validate_load=validate)
    with pytest.raises(ToolExecutionError, match='fixed context'):
        mandatory.initialize()
    assert mandatory._read() == {}
