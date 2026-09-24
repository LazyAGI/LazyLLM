from types import SimpleNamespace

import pytest

from lazyllm.tools import (
    AuthorizationDecision, HostFileAccess, HostFileIntent, HostFileResolution,
    ToolExecutionDisposition, ToolManager, fc_register,
)


def call(tool_name='tool', **arguments):
    return {'function': {'name': tool_name, 'arguments': arguments}}


def make_tool(capability, tmp_path, **metadata):
    def tool(value: int):
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    if capability in ('read', 'write', 'delete'):
        metadata['host_file'] = lambda args: HostFileResolution(
            args, (HostFileIntent(str(tmp_path / 'file'), capability),))
    elif capability != 'UNDECLARED':
        metadata['host_file'] = capability
    return fc_register(execute_in_sandbox=False, **metadata)(tool)


@pytest.mark.parametrize('capability', ['UNDECLARED', 'NONE', 'OPAQUE', 'read', 'write', 'delete'])
def test_default_mode_executes_every_ready_tool(capability, tmp_path):
    manager = ToolManager([make_tool(capability, tmp_path)])
    assert manager(call(value=7))[0] == {'ok': True, 'value': 7}


@pytest.mark.parametrize('decision', list(AuthorizationDecision))
def test_explicit_policy_and_mixed_preparation_failures(decision, tmp_path):
    manager = ToolManager([make_tool('UNDECLARED', tmp_path)])
    prepared = manager.prepare_tool_calls([call(value='invalid'), call(value=7)],
                                          authorization_policy=lambda _: decision)
    assert not prepared[0].ready
    assert prepared[1].authorization is decision
    if decision is AuthorizationDecision.DENY:
        with pytest.raises(ValueError, match='DENY'):
            manager.execute_prepared(prepared, selected_indices=(0, 1))
        return
    approved = (1,) if decision is AuthorizationDecision.ASK else ()
    if approved:
        with pytest.raises(ValueError, match='unapproved ASK'):
            manager.execute_prepared(prepared, selected_indices=(0, 1))
    result = manager.execute_prepared(prepared, selected_indices=(0, 1), approved_indices=approved)
    assert result.records[0].disposition is ToolExecutionDisposition.PREPARATION_FAILED
    assert result.results[1] == {'ok': True, 'value': 7}


def test_selector_cannot_approve_calls(tmp_path, monkeypatch):
    manager = ToolManager([make_tool('UNDECLARED', tmp_path)])
    prepared = manager.prepare_tool_calls(call(value=7), authorization_policy=lambda _: AuthorizationDecision.ASK)
    monkeypatch.setattr(manager, 'prepare_tool_calls', lambda *a, **kw: prepared)
    with pytest.raises(ValueError, match='unapproved ASK'):
        manager.execute_with_records(call(value=7), dispatch_selector=lambda calls: (0,))


def test_mcp_adapter_marks_source_without_requiring_server_metadata():
    from lazyllm.tools.mcp.tool_adaptor import generate_lazyllm_tool

    calls = []

    class Client:
        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return SimpleNamespace(content=[])

    remote = SimpleNamespace(name='remote.echo', description='Echo a value.', inputSchema={
        'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value'],
    })
    manager = ToolManager([generate_lazyllm_tool(Client(), remote)])
    prepared = manager.prepare_tool_calls(call('remote_echo', value=7))
    assert prepared[0].tool_source == 'mcp'
    assert prepared[0].host_file_access is HostFileAccess.UNDECLARED
    assert manager.execute_prepared(prepared).results[0]['ok']
    assert calls == [('remote.echo', {'value': 7})]


def test_mixed_selection_checks_authorization_before_any_effect():
    effects = []

    @fc_register(execute_in_sandbox=False)
    def record(value: int):
        '''Record a value.

        Args:
            value: Input value.
        '''
        effects.append(value)
        return value

    class Policy:
        def __bool__(self):
            return False

        def decide(self, prepared):
            return (AuthorizationDecision.ALLOW if prepared.validated_arguments['value'] == 1
                    else AuthorizationDecision.ASK)

    manager = ToolManager([record])
    prepared = manager.prepare_tool_calls([call('record', value=1), call('record', value=2)],
                                          authorization_policy=Policy())
    with pytest.raises(ValueError, match='unapproved ASK'):
        manager.execute_prepared(prepared, selected_indices=(0, 1))
    assert effects == []
    manager.execute_prepared(prepared, selected_indices=(0, 1), approved_indices=(1,))
    assert sorted(effects) == [1, 2]
