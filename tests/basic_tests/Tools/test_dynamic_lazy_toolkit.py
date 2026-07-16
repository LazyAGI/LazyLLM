from lazyllm.tools.agent.toolsManager import ToolManager


class _DynamicToolkit:
    '''Toolkit used to verify request-local lazy registration.'''

    __public_apis__ = ['search']

    def __init__(self, lazy: bool):
        self.lazy = lazy

    def __lazy_source__(self) -> bool:
        return self.lazy

    def search(self, query: str) -> str:
        '''Search for a query.

        Args:
            query: Query text.

        Returns:
            Search result.
        '''
        return query


def _description_names(manager: ToolManager) -> list[str]:
    return [item['function']['name'] for item in manager.tools_description]


def test_dynamic_lazy_toolkit_uses_gateway_when_source_is_truthy():
    manager = ToolManager([_DynamicToolkit(lazy=True)])

    assert _description_names(manager) == ['get__DynamicToolkit_methods']


def test_dynamic_lazy_toolkit_exposes_methods_when_source_is_falsy():
    manager = ToolManager([_DynamicToolkit(lazy=False)])

    assert _description_names(manager) == ['_DynamicToolkit_search']
