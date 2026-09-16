from types import SimpleNamespace

import pytest

from lazyllm import config
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


@pytest.mark.parametrize('capability, expected', [
    ('UNDECLARED', AuthorizationDecision.DENY),
    ('NONE', AuthorizationDecision.ALLOW),
    ('OPAQUE', AuthorizationDecision.ASK),
    ('read', AuthorizationDecision.ALLOW),
    ('write', AuthorizationDecision.ASK),
    ('delete', AuthorizationDecision.ASK),
])
def test_security_mode_requires_explicit_opt_in(capability, expected, tmp_path):
    manager = ToolManager([make_tool(capability, tmp_path)])
    with config.temp('host_file_security_enabled', True):
        prepared = manager.prepare_tool_calls(call(value=7))
    assert prepared[0].authorization is expected
    result = manager.execute_prepared(prepared)
    assert result.results[0]['ok'] is (expected is AuthorizationDecision.ALLOW)
    if expected is AuthorizationDecision.ASK:
        assert result.records[0].reason == 'approval_required'
        assert manager.execute_prepared(prepared, approved_indices=(0,)).results[0]['value'] == 7
    elif expected is AuthorizationDecision.DENY:
        assert result.records[0].reason == 'authorization_rejected'


@pytest.mark.parametrize('capability', ['UNDECLARED', 'NONE', 'OPAQUE', 'read', 'write', 'delete'])
def test_full_trust_allows_ready_calls_in_security_mode(capability, tmp_path):
    manager = ToolManager([make_tool(capability, tmp_path)])
    with config.temp('host_file_security_enabled', True), config.temp('host_file_full_trust', True):
        assert manager(call(value=7))[0] == {'ok': True, 'value': 7}


@pytest.mark.parametrize('security, full_trust', [(False, False), (True, False), (True, True)])
def test_modes_preserve_preparation_failures_and_explicit_policy(security, full_trust, tmp_path):
    def broken_resolver(args):
        raise ValueError('cannot resolve')

    broken = make_tool('UNDECLARED', tmp_path, host_file=broken_resolver)
    broken.__name__ = 'broken'
    manager = ToolManager([make_tool('UNDECLARED', tmp_path), broken])
    with config.temp('host_file_security_enabled', security), config.temp('host_file_full_trust', full_trust):
        prepared = manager.prepare_tool_calls([
            call(value='invalid'), call('missing'), call('broken', value=7), call(value=7),
        ], authorization_policy=lambda _: AuthorizationDecision.ALLOW)
        assert [item.authorization for item in prepared] == [AuthorizationDecision.DENY] * 3 + [
            AuthorizationDecision.ALLOW]
        results = manager.execute_prepared(prepared)
        assert [item.disposition for item in results.records] == [ToolExecutionDisposition.PREPARATION_FAILED] * 3 + [
            ToolExecutionDisposition.EXECUTED]
        denied = manager.prepare_tool_calls(call(value=7), authorization_policy=lambda _: AuthorizationDecision.DENY)
        assert not manager.execute_prepared(denied).results[0]['ok']


@pytest.mark.parametrize('source', ['mcp', 'skill'])
def test_undeclared_external_tools_remain_usable_in_security_mode(source, tmp_path):
    manager = ToolManager([make_tool('UNDECLARED', tmp_path, tool_source=source)])
    with config.temp('host_file_security_enabled', True):
        prepared = manager.prepare_tool_calls(call(value=7))
    assert prepared[0].tool_source == source
    assert prepared[0].host_file_access is HostFileAccess.UNDECLARED
    assert manager.execute_prepared(prepared).results[0]['value'] == 7
    assert 'tool_source' not in str(manager.tools_description)


@pytest.mark.parametrize('source', ['mcp', 'skill'])
def test_external_source_does_not_bypass_explicit_declarations(source, tmp_path):
    manager = ToolManager([make_tool('OPAQUE', tmp_path, tool_source=source)])
    with config.temp('host_file_security_enabled', True):
        prepared = manager.prepare_tool_calls(call(value=7))
    assert prepared[0].authorization is AuthorizationDecision.ASK
    assert not manager.execute_prepared(prepared).results[0]['ok']


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
    with config.temp('host_file_security_enabled', True):
        prepared = manager.prepare_tool_calls(call('remote_echo', value=7))
    assert prepared[0].tool_source == 'mcp'
    assert prepared[0].host_file_access is HostFileAccess.UNDECLARED
    assert manager.execute_prepared(prepared).results[0]['ok']
    assert calls == [('remote.echo', {'value': 7})]


def test_skill_adapter_marks_source_and_preserves_declared_policy(tmp_path):
    from lazyllm.tools.agent import SkillManager

    manager = ToolManager(SkillManager(dir=str(tmp_path)).get_skill_tools())
    assert {tool.runtime_metadata.tool_source for tool in manager.all_tools} == {'skill'}
    with config.temp('host_file_security_enabled', True):
        prepared = manager.prepare_tool_calls(call('run_script', name='demo', rel_path='scripts/run.py'))
    assert prepared[0].tool_source == 'skill'
    assert prepared[0].authorization is AuthorizationDecision.ASK


def test_config_defaults_and_environment(monkeypatch):
    assert config['host_file_security_enabled'] is False
    assert config['host_file_full_trust'] is False
    with config.temp('host_file_security_enabled', False), config.temp('host_file_full_trust', False):
        monkeypatch.setenv('LAZYLLM_HOST_FILE_SECURITY_ENABLED', 'true')
        monkeypatch.setenv('LAZYLLM_HOST_FILE_FULL_TRUST', 'true')
        config.refresh(['host_file_security_enabled', 'host_file_full_trust'])
        assert config['host_file_security_enabled'] is True
        assert config['host_file_full_trust'] is True
