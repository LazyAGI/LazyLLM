from dataclasses import FrozenInstanceError

import lazyllm
import pytest

from lazyllm.tools import ToolManager, ToolRuntimeMetadata, fc_register


def _documented_tool(name='runtime_contract_tool'):
    def tool(value: str) -> str:
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    tool.__name__ = name
    return tool


def test_runtime_metadata_validation_and_resolution():
    with pytest.raises(ValueError, match='exclusive cannot be combined'):
        ToolRuntimeMetadata(exclusive=True, read_keys='resource')
    with pytest.raises(TypeError, match='polling must be a bool'):
        ToolRuntimeMetadata(polling='yes')
    with pytest.raises(ValueError, match='resource keys must not contain empty'):
        ToolRuntimeMetadata(read_keys='')

    metadata = ToolRuntimeMetadata(
        read_keys=['shared', ('file', '.')],
        write_keys='shared',
        polling=True,
    )
    access = metadata.resolve({})
    assert not access.read_keys.intersection(access.write_keys)
    assert access.write_keys
    assert access.polling is True
    assert access.counts_as_progress is True
    with pytest.raises(FrozenInstanceError):
        metadata.polling = False


def test_metadata_only_registration_and_instance_isolation():
    tool = _documented_tool()
    fc_register(read_keys='shared')(tool)
    manager = ToolManager([tool])
    first = manager.all_tools[0]
    second = ToolManager([tool]).all_tools[0]

    assert first.runtime_metadata is second.runtime_metadata
    assert first._resolve_runtime_access({'value': 'x'}).read_keys
    first.output_files = ['first.txt']
    copied = first.output_files
    copied.append('copy-only.txt')
    assert first.output_files == ['first.txt']
    assert second.output_files == []


def test_registration_forms_and_compatible_alias():
    bare = _documented_tool('runtime_bare_tool')
    grouped = _documented_tool('runtime_grouped_tool')
    aliased = _documented_tool('runtime_alias_tool')
    try:
        assert fc_register(bare) is bare
        assert fc_register('tool', execute_in_sandbox=False)(grouped) is grouped
        from lazyllm.tools.agent import register
        assert register is fc_register
        assert register('tool', polling=True)(aliased) is aliased

        assert lazyllm.tool.runtime_bare_tool().execute_in_sandbox is True
        assert lazyllm.tool.runtime_grouped_tool().execute_in_sandbox is False
        assert lazyllm.tool.runtime_alias_tool().runtime_metadata.polling is True
    finally:
        for name in ('runtime_bare_tool', 'runtime_grouped_tool', 'runtime_alias_tool'):
            lazyllm.tool.remove(name)


def test_stacked_groups_reuse_one_declaration_and_conflicts_fail():
    tool = _documented_tool('runtime_stacked_tool')

    def read_keys(args):
        return ('file', args['value'])

    try:
        fc_register('tool', read_keys=read_keys)(tool)
        fc_register('builtin_tools', read_keys=read_keys)(tool)
        assert lazyllm.tool.runtime_stacked_tool().runtime_metadata.read_keys is read_keys
        assert lazyllm.builtin_tools.runtime_stacked_tool().runtime_metadata.read_keys is read_keys

        with pytest.raises(ValueError, match='conflicting ToolRuntimeMetadata'):
            fc_register(write_keys=lambda args: ('file', args['value']))(tool)
    finally:
        lazyllm.tool.remove('runtime_stacked_tool')
        lazyllm.builtin_tools.remove('runtime_stacked_tool')


def test_method_metadata_preserves_non_sandbox_execution():
    class Toolkit:
        __public_apis__ = ['read']

        @fc_register(read_keys='shared')
        def read(self, value: str) -> str:
            '''Read a value.

            Args:
                value: Input value.
            '''
            return value

    tool = ToolManager([Toolkit()]).tools_info['Toolkit_read']
    assert tool.execute_in_sandbox is False
    assert tool._resolve_runtime_access({'value': 'x'}).read_keys


def test_resolve_tool_accesses_returns_neutral_for_invalid_calls():
    tool = _documented_tool()
    fc_register(write_keys='shared')(tool)
    manager = ToolManager([tool])

    accesses = manager.resolve_tool_accesses([
        {'id': 'bad', 'function': {'name': 'missing', 'arguments': '{}'}},
        {'id': 'good', 'function': {'name': tool.__name__, 'arguments': '{"value":"x"}'}},
    ])

    assert not accesses[0].read_keys and not accesses[0].write_keys and not accesses[0].exclusive
    assert accesses[1].write_keys
