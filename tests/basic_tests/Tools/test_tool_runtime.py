from dataclasses import FrozenInstanceError

import lazyllm
import pytest

from lazyllm.tools import (
    PreparedToolBatch,
    ToolExecutionDisposition,
    ToolManager,
    ToolRuntimeMetadata,
    fc_register,
)


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
    assert not hasattr(access, 'counts_as_progress')
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


def test_stacked_metadata_merges_fields_idempotently_and_rejects_conflicts():
    tool = _documented_tool('runtime_stacked_tool')

    def read_keys(args):
        return ('file', args['value'])

    fc_register(read_keys=read_keys)(tool)
    fc_register(polling=True)(tool)
    fc_register(polling=True)(tool)
    metadata = ToolManager([tool]).all_tools[0].runtime_metadata
    assert metadata.read_keys is read_keys
    assert metadata.polling is True

    with pytest.raises(ValueError, match='conflicting ToolRuntimeMetadata'):
        fc_register(read_keys='other-resource')(tool)


def test_metadata_patch_after_registration_updates_future_instances():
    tool = _documented_tool('runtime_late_patch_tool')
    try:
        fc_register('tool', read_keys='shared')(tool)
        fc_register(polling=True)(tool)

        metadata = lazyllm.tool.runtime_late_patch_tool().runtime_metadata
        assert metadata.read_keys == 'shared'
        assert metadata.polling is True
    finally:
        lazyllm.tool.remove('runtime_late_patch_tool')


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


def test_prepared_calls_expose_access_and_invalid_preparation_state():
    tool = _documented_tool()
    fc_register(write_keys='shared')(tool)
    manager = ToolManager([tool])

    prepared = manager.prepare_tool_calls([
        {'id': 'bad', 'function': {'name': 'missing', 'arguments': '{}'}},
        {'id': 'good', 'function': {'name': tool.__name__, 'arguments': '{"value":"x"}'}},
    ])

    assert isinstance(prepared, PreparedToolBatch)
    assert prepared[0].preparation_status == 'invalid'
    assert not prepared[0].access.read_keys and not prepared[0].access.write_keys
    assert prepared[1].preparation_status == 'ready'
    assert prepared[1].access.write_keys
    assert not hasattr(manager, 'resolve_tool_accesses')

    batch = manager.execute_prepared_calls(prepared)
    assert [record.executed for record in batch.records] == [False, True]
    assert [record.disposition for record in batch.records] == [
        ToolExecutionDisposition.PREPARATION_FAILED,
        ToolExecutionDisposition.EXECUTED,
    ]


def test_prepared_batch_is_manager_owned_single_use_and_views_are_not_execution_inputs():
    seen = []

    @fc_register(write_keys=lambda args: ('file', args['value']))
    def tool(value: str) -> str:
        '''Return a value.

        Args:
            value: Input value.
        '''
        seen.append(value)
        return value

    first = ToolManager([tool])
    second = ToolManager([tool])
    prepared = first.prepare_tool_calls({
        'id': 'one',
        'function': {'name': 'tool', 'arguments': {'value': 'original'}},
    })

    prepared[0].validated_arguments['value'] = 'forged'
    prepared[0].tool_call['function']['arguments'] = '{"value":"forged"}'
    assert prepared[0].validated_arguments == {'value': 'original'}
    with pytest.raises(TypeError, match='created by ToolManager'):
        PreparedToolBatch((), (), first)
    with pytest.raises(ValueError, match='different ToolManager'):
        second.execute_prepared_calls(prepared)

    batch = first.execute_prepared_calls(prepared)
    assert list(batch.results) == [{'ok': True, 'value': 'original'}]
    assert batch.records[0].validated_arguments == {'value': 'original'}
    assert seen == ['original']
    with pytest.raises(RuntimeError, match='already been executed'):
        first.execute_prepared_calls(prepared)


def test_dispatch_failure_has_explicit_disposition():
    tool = _documented_tool('runtime_dispatch_tool')
    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls({
        'id': 'one',
        'function': {'name': tool.__name__, 'arguments': {'value': 'x'}},
    })
    manager._tool_call.pop(tool.__name__)

    batch = manager.execute_prepared_calls(prepared)

    assert batch.records[0].disposition is ToolExecutionDisposition.DISPATCH_FAILED
    assert batch.records[0].executed is False
    assert batch.results[0]['ok'] is False


def test_prepared_batch_rejects_duplicate_selected_indices():
    tool = _documented_tool('runtime_selected_tool')
    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls({
        'id': 'one',
        'function': {'name': tool.__name__, 'arguments': {'value': 'x'}},
    })

    with pytest.raises(ValueError, match='must be unique'):
        manager.execute_prepared_calls(prepared, selected_indices=(0, 0))


def test_prepare_and_execute_validate_resolve_and_invoke_once():
    calls = {'validate': 0, 'resolve': 0, 'invoke': 0}

    def resolve(arguments):
        calls['resolve'] += 1
        return f'resource:{arguments["value"]}'

    @fc_register(read_keys=resolve)
    def tool(value: str) -> str:
        '''Return a value.

        Args:
            value: Input value.
        '''
        calls['invoke'] += 1
        return value

    manager = ToolManager([tool])
    module_tool = manager.all_tools[0]
    original_validate = module_tool._validate_input

    def validate(arguments):
        calls['validate'] += 1
        return original_validate(arguments)

    module_tool._validate_input = validate
    batch = manager.execute_with_records({
        'id': 'one',
        'function': {'name': 'tool', 'arguments': '{"value":"x",}'},
    })

    assert calls == {'validate': 1, 'resolve': 1, 'invoke': 1}
    assert list(batch.results) == [{'ok': True, 'value': 'x'}]
    assert batch.records[0].arguments == {'value': 'x'}
    assert batch.records[0].validated_arguments == {'value': 'x'}
    assert batch.records[0].executed is True
