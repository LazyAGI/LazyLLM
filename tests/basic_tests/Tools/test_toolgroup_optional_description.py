import pytest

from lazyllm.tools.agent.toolsManager import ToolManager


def increment(value: int) -> int:
    '''Increment a number.

    Args:
        value (int): Number to increment.
    '''
    return value + 1


@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('description', [None, '', 'Number tools'])
def test_optional_dict_description_matches_direct_group(lazy, description):
    group = dict(name='numbers', tools=[increment], lazy=lazy)
    if description is not None:
        group['desc'] = description
    manager = ToolManager([group])
    descriptions = manager.tools_description
    names = [item['function']['name'] for item in descriptions]
    assert names == (['get_numbers_methods'] if lazy else ['increment'])
    if lazy:
        assert (description or '') in descriptions[0]['function']['description']
    assert manager._tools[0].get_flat_tools()['increment']({'value': 2}) == 3


def test_nested_dict_groups_allow_omitted_descriptions():
    manager = ToolManager([dict(name='outer', lazy=False, tools=[
        dict(name='inner', lazy=False, tools=[increment]),
    ])])
    assert [item['function']['name'] for item in manager.tools_description] == ['increment']
    assert manager._tools[0].get_flat_tools()['increment']({'value': 4}) == 5
