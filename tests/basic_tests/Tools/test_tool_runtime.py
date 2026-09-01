from dataclasses import FrozenInstanceError

import lazyllm
import pytest

from lazyllm.tools import (
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
    assert metadata.polling is True
    assert not hasattr(access, 'polling')
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


def test_dispatch_selector_sees_prepared_snapshots_and_failures_are_recorded():
    tool = _documented_tool()
    fc_register(write_keys='shared', polling=True)(tool)
    manager = ToolManager([tool])
    selected = []

    def selector(prepared):
        selected.extend(prepared)
        return range(len(prepared))

    batch = manager.execute_with_records(
        [
            {'id': 'bad', 'function': {'name': 'missing', 'arguments': '{}'}},
            {'id': 'good', 'function': {'name': tool.__name__, 'arguments': '{"value":"x"}'}},
        ],
        dispatch_selector=selector,
    )

    assert [item.index for item in selected] == [0, 1]
    assert selected[0].ready is False
    assert not selected[0].access.read_keys and not selected[0].access.write_keys
    assert selected[1].ready is True
    assert selected[1].access.write_keys
    assert selected[1].polling is True
    assert not hasattr(manager, 'resolve_tool_accesses')
    assert [record.disposition for record in batch.records] == [
        ToolExecutionDisposition.PREPARATION_FAILED,
        ToolExecutionDisposition.EXECUTED,
    ]


def test_dispatch_selector_snapshot_is_not_an_execution_input():
    seen = []

    @fc_register(write_keys=lambda args: ('file', args['value']))
    def tool(value: str) -> str:
        '''Return a value.

        Args:
            value: Input value.
        '''
        seen.append(value)
        return value

    def selector(prepared):
        prepared[0].validated_arguments['value'] = 'forged'
        prepared[0].tool_call['function']['arguments'] = '{"value":"forged"}'
        return (0,)

    batch = ToolManager([tool]).execute_with_records(
        {
            'id': 'one',
            'function': {'name': 'tool', 'arguments': {'value': 'original'}},
        },
        dispatch_selector=selector,
    )
    assert list(batch.results) == [{'ok': True, 'value': 'original'}]
    assert seen == ['original']


def test_dispatch_selector_uses_original_order_for_selected_calls():
    tool = _documented_tool('runtime_selected_tool')
    calls = [
        {'id': str(index), 'function': {
            'name': tool.__name__, 'arguments': {'value': str(index)},
        }}
        for index in range(3)
    ]

    batch = ToolManager([tool]).execute_with_records(
        calls,
        dispatch_selector=lambda _prepared: (2, 0),
    )

    assert [record.index for record in batch.records] == [0, 2]
    assert [result['value'] for result in batch.results] == ['0', '2']


@pytest.mark.parametrize(
    ('indices', 'error', 'message'),
    [
        ((0, 0), ValueError, 'must be unique'),
        ((1,), IndexError, 'out of range'),
        ((True,), IndexError, 'out of range'),
    ],
)
def test_dispatch_selector_rejects_invalid_indices(indices, error, message):
    tool = _documented_tool('runtime_selected_tool')
    manager = ToolManager([tool])
    tool_call = {
        'id': 'one',
        'function': {'name': tool.__name__, 'arguments': {'value': 'x'}},
    }

    with pytest.raises(error, match=message):
        manager.execute_with_records(
            tool_call,
            dispatch_selector=lambda _prepared: indices,
        )


def test_empty_dispatch_selection_does_not_invoke_tool():
    calls = []

    def tool(value: str) -> str:
        '''Return a value.

        Args:
            value: Input value.
        '''
        calls.append(value)
        return value

    batch = ToolManager([tool]).execute_with_records(
        {'id': 'one', 'function': {'name': 'tool', 'arguments': {'value': 'x'}}},
        dispatch_selector=lambda _prepared: (),
    )

    assert batch.results == []
    assert batch.records == ()
    assert calls == []


def test_dispatch_selector_failure_does_not_invoke_tool():
    calls = []

    def tool(value: str) -> str:
        '''Return a value.

        Args:
            value: Input value.
        '''
        calls.append(value)
        return value

    def fail(_prepared):
        raise RuntimeError('selector failed')

    with pytest.raises(RuntimeError, match='selector failed'):
        ToolManager([tool]).execute_with_records(
            {'id': 'one', 'function': {'name': 'tool', 'arguments': {'value': 'x'}}},
            dispatch_selector=fail,
        )
    assert calls == []


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
    assert batch.records[0].disposition is ToolExecutionDisposition.EXECUTED
