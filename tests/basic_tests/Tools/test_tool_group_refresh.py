import copy

import lazyllm
import pytest
from lazyllm.tools.agent.toolsManager import ToolManager


def lookup(query: str) -> str:
    '''Look up a document.

    Args:
        query (str): Search text.
    '''
    return query


def read_document(document_id: str) -> str:
    '''Read a document.

    Args:
        document_id (str): Document identifier.
    '''
    return document_id


@pytest.fixture(autouse=True)
def isolated_state():
    old = lazyllm.locals['_lazyllm_agent']
    auth = copy.deepcopy(lazyllm.globals.config['dynamic_tool_auth'])
    lazyllm.locals['_lazyllm_agent'] = {'workspace': {}}
    yield
    lazyllm.locals['_lazyllm_agent'] = old
    lazyllm.globals.config['dynamic_tool_auth'] = auth


class Store:
    def __init__(self):
        self.state = {'loaded': [], 'skills': {}}
        self.fail = False

    def read(self):
        return copy.deepcopy(self.state)

    def update(self, update):
        state = update(self.read())
        if self.fail:
            raise OSError('disk full')
        self.state = copy.deepcopy(state)
        return state


@pytest.mark.parametrize('retrieval', [False, True])
def test_refresh_nested_group_publishes_new_members_and_load_state(retrieval):
    manager = ToolManager([{'name': 'outer', 'desc': 'outer', 'prefix': False, 'tools': [
        {'name': 'docs', 'desc': 'Documents', 'prefix': False, 'tools': [lookup]}]}])
    if retrieval:
        manager.enable_tool_retrieval(groups={'docs'}, required=[], threshold_tokens=100,
                                      estimate_tokens=lambda _: 1)
    result = manager.refresh_tool_group('docs', definition={
        'name': 'docs', 'desc': 'Documents', 'prefix': False, 'tools': [read_document]}, load=True)
    assert result['status'] == 'ready'
    assert 'read_document' in {d['function']['name'] for d in manager.tools_description}
    assert 'lookup' not in manager.tools_info
    assert manager.get_tool_group_state('docs')['loaded'] is True


@pytest.mark.parametrize('failure', ['budget', 'store'])
def test_failed_refresh_keeps_catalog_auth_and_durable_state(failure):
    manager = ToolManager([{'name': 'docs', 'desc': 'Documents', 'prefix': False, 'tools': [lookup]}])
    store = Store()
    manager.group_state_store = store
    manager.activate_group('docs')
    original = copy.deepcopy(manager.tools_description)
    lazyllm.globals.config['dynamic_tool_auth'] = {'bing': 'old', 'google': 'keep'}
    if failure == 'budget':
        manager.tool_load_validator = lambda _: (_ for _ in ()).throw(ValueError('budget'))
    else:
        store.fail = True
    result = manager.refresh_tool_group('docs', definition={
        'name': 'docs', 'desc': 'Documents', 'prefix': False, 'tools': [read_document]},
        tool_config={'bing': 'new'}, load=True)
    assert result['status'] == ('budget_blocked' if failure == 'budget' else 'unavailable')
    assert manager.tools_description == original
    assert store.read()['loaded'] == ['docs']
    assert lazyllm.globals.config['dynamic_tool_auth'] == {'bing': 'old', 'google': 'keep'}


def test_revoked_group_stays_blocked_after_failed_refresh():
    manager = ToolManager([{'name': 'docs', 'desc': 'Documents', 'prefix': False, 'lazy': False, 'tools': [lookup]}])
    manager.refresh_tool_group('docs', available=False)
    manager.tool_load_validator = lambda _: (_ for _ in ()).throw(ValueError('budget'))
    assert manager.refresh_tool_group('docs', load=True)['status'] == 'budget_blocked'
    assert not manager.get_tool_group_state('docs')['available']
    assert 'lookup' not in {d['function']['name'] for d in manager.tools_description}
    result = manager([{'id': 'old', 'function': {'name': 'lookup', 'arguments': '{"query":"x"}'}}],
                     allowed_tool_names={'lookup'})
    assert result[0]['ok'] is False


def test_config_clear_is_scoped_and_unknown_group_is_invalid():
    manager = ToolManager([{'name': 'docs', 'desc': 'Documents', 'tools': [lookup]}])
    lazyllm.globals.config['dynamic_tool_auth'] = {'bing': 'old', 'google': 'keep'}
    assert manager.refresh_tool_group('docs', tool_config={'bing': None})['status'] == 'ready'
    assert lazyllm.globals.config['dynamic_tool_auth'] == {'google': 'keep'}
    with pytest.raises(ValueError):
        manager.refresh_tool_group('missing')


def test_legacy_replace_can_clear_a_group_and_preserves_validation_exception():
    manager = ToolManager([{'name': 'docs', 'desc': 'Documents', 'tools': [lookup]}])
    manager.replace_tool_group('docs', {'name': 'docs', 'desc': 'Documents', 'tools': []})
    assert 'lookup' not in manager.tools_info
    manager.tool_load_validator = lambda _: (_ for _ in ()).throw(ValueError('original validation error'))
    with pytest.raises(ValueError, match='original validation error'):
        manager.replace_tool_group('docs', {'name': 'docs', 'desc': 'Documents', 'tools': [lookup]})


def test_nested_native_credentials_are_invalidated_before_readiness_check():
    from lazyllm.common.credential_mixin import CredentialMixin
    from lazyllm.common.auth import Credential
    from lazyllm.tools.agent.toolsManager import InstanceToolGroup

    class NativeDocuments(CredentialMixin):
        '''Documents with dynamic credentials.'''
        __public_apis__ = ['read']

        def __init__(self):
            self.__init_credential__(Credential(kind='dynamic'))

        def _resolve_dynamic_token(self):
            return (lazyllm.globals.config['dynamic_tool_auth'] or {}).get('documents', '')

        def read(self, query: str) -> str:
            '''Read using the configured account.

            Args:
                query (str): Document query.
            '''
            return self.get_current_token()

    instance = NativeDocuments()
    manager = ToolManager([{'name': 'outer', 'desc': 'outer', 'prefix': False, 'tools': [
        {'name': 'inner', 'desc': 'inner', 'prefix': False, 'tools': [InstanceToolGroup(instance)]}]}])
    lazyllm.locals['curr_key'][instance._credential_id] = 'stale'
    lazyllm.globals.config['dynamic_tool_auth'] = {'documents': 'old', 'google': 'keep'}
    try:
        assert manager.refresh_tool_group('outer', tool_config={'documents': 'new'})['status'] == 'ready'
        assert instance.get_current_token() == 'new'
        manager.refresh_tool_group('outer', tool_config={'documents': '   '})
        assert 'documents' not in lazyllm.globals.config['dynamic_tool_auth']
    finally:
        lazyllm.locals['curr_key'].pop(instance._credential_id, None)


def test_refresh_nested_eager_group_preserves_prefix_gateway_and_activation():
    manager = ToolManager([{'name': 'outer', 'desc': 'outer', 'prefix': True, 'tools': [
        {'name': 'inner', 'desc': 'inner', 'prefix': True, 'lazy': False, 'tools': [lookup]}]}])
    result = manager.refresh_tool_group('inner', definition={
        'name': 'inner', 'desc': 'inner', 'prefix': True, 'lazy': False, 'tools': [read_document]}, load=True)
    assert result['status'] == 'ready'
    assert manager.get_tool_group_state('inner')['members'] == ['outer_inner_read_document']
    assert manager.get_tool_group_state('inner')['loaded']
    gateway = manager._find_tool_group('outer')._gateway_tool
    assert 'outer_inner_read_document' in gateway({})
    assert 'outer_inner_lookup' not in gateway({})


def test_cancelled_refresh_restores_revocation_credentials_and_catalog():
    import asyncio

    manager = ToolManager([{'name': 'docs', 'desc': 'Documents', 'prefix': False, 'tools': [lookup]}])
    lazyllm.globals.config['dynamic_tool_auth'] = {'bing': 'old'}
    manager.refresh_tool_group('docs', available=False)

    def cancel(_):
        raise asyncio.CancelledError()

    manager.tool_load_validator = cancel
    with pytest.raises(asyncio.CancelledError):
        manager.refresh_tool_group('docs', definition={
            'name': 'docs', 'desc': 'Documents', 'prefix': False, 'tools': [read_document]},
            tool_config={'bing': 'new'}, load=True)
    assert not manager.get_tool_group_state('docs')['available']
    assert 'read_document' not in manager.tools_info
    assert lazyllm.globals.config['dynamic_tool_auth'] == {'bing': 'old'}


def test_clearing_required_credential_never_restores_old_permission():
    class Documents:
        __public_apis__ = ['lookup']
        lookup = staticmethod(lookup)

    manager = ToolManager([{'name': 'docs', 'desc': 'Documents', 'pick_first_valid': True,
                            'tools': [(Documents(), lambda _: (
                                lazyllm.globals.config['dynamic_tool_auth'] or {}).get('bing'))]}])
    lazyllm.globals.config['dynamic_tool_auth'] = {'bing': 'old', 'google': 'keep'}
    manager.refresh_tool_group('docs', tool_config={'bing': None})
    assert lazyllm.globals.config['dynamic_tool_auth'] == {'google': 'keep'}
    assert not manager.get_tool_group_state('docs')['available']


def test_nested_state_respects_revoked_parent():
    manager = ToolManager([{'name': 'outer', 'desc': 'outer', 'tools': [
        {'name': 'inner', 'desc': 'inner', 'lazy': False, 'tools': [lookup]}]}])
    manager.refresh_tool_group('outer', available=False)
    assert not manager.get_tool_group_state('inner')['available']
