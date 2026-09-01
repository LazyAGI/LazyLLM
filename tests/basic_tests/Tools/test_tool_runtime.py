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


def test_runtime_metadata_registration_contract():
    tool = _documented_tool()
    from lazyllm.tools.agent import register

    assert register is fc_register
    fc_register(read_keys=['shared', 'read-only'])(tool)
    fc_register(write_keys='shared')(tool)
    fc_register(polling=True)(tool)
    fc_register(polling=True)(tool)

    metadata = ToolManager([tool]).all_tools[0].runtime_metadata
    access = metadata.resolve({})
    assert metadata.polling is True
    assert access.read_keys
    assert access.write_keys
    assert not access.read_keys.intersection(access.write_keys)
    with pytest.raises(ValueError, match='conflicting ToolRuntimeMetadata'):
        fc_register(read_keys='other-resource')(tool)
    with pytest.raises(ValueError, match='exclusive cannot be combined'):
        ToolRuntimeMetadata(exclusive=True, read_keys='resource')


def test_execute_with_records_prepares_once_and_records_failure():
    calls = {'validate': 0, 'resolve': 0, 'invoke': 0}
    snapshots = []

    def resolve(arguments):
        calls['resolve'] += 1
        return f'resource:{arguments["value"]}'

    @fc_register(read_keys=resolve, polling=True)
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

    def select(prepared):
        snapshots.extend(prepared)
        return range(len(prepared))

    batch = manager.execute_with_records(
        [
            {'id': 'bad', 'function': {'name': 'missing', 'arguments': '{}'}},
            {'id': 'good', 'function': {'name': 'tool', 'arguments': '{"value":"x",}'}},
        ],
        dispatch_selector=select,
    )

    assert calls == {'validate': 1, 'resolve': 1, 'invoke': 1}
    assert [item.index for item in snapshots] == [0, 1]
    assert snapshots[0].ready is False
    assert snapshots[1].ready is True
    assert snapshots[1].validated_arguments == {'value': 'x'}
    assert snapshots[1].access.read_keys
    assert snapshots[1].polling is True
    assert [record.disposition for record in batch.records] == [
        ToolExecutionDisposition.PREPARATION_FAILED,
        ToolExecutionDisposition.EXECUTED,
    ]
    assert batch.results[0]['ok'] is False
    assert batch.results[1] == {'ok': True, 'value': 'x'}


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


def test_dispatch_selector_preserves_original_order():
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
    ('indices', 'error'),
    [
        ((0, 0), ValueError),
        ((1,), IndexError),
    ],
)
def test_dispatch_selector_rejects_invalid_indices(indices, error):
    tool = _documented_tool('runtime_selected_tool')
    manager = ToolManager([tool])
    tool_call = {
        'id': 'one',
        'function': {'name': tool.__name__, 'arguments': {'value': 'x'}},
    }

    with pytest.raises(error):
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
