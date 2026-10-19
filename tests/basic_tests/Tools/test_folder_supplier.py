import pytest

from lazyllm.tools.fs.supplier.feishu import FeishuFS, FeishuWikiFS
from lazyllm.tools.fs.supplier.googledrive import GoogleDriveFS


@pytest.mark.parametrize('factory', [GoogleDriveFS, FeishuFS, FeishuWikiFS])
def test_bound_account_credential_is_instance_local(factory):
    first = factory(dynamic_auth=True, skip_instance_cache=True)
    second = factory(dynamic_auth=True, skip_instance_cache=True)
    first.bind_access_token('account-a')
    second.bind_access_token('account-b')
    assert first.get_current_token() == 'account-a'
    assert second.get_current_token() == 'account-b'
    assert first.inject_auth_header()['Authorization'] == 'Bearer account-a'
    assert second.inject_auth_header()['Authorization'] == 'Bearer account-b'
    with pytest.raises(ValueError):
        first.bind_access_token('')
    assert first.get_current_token() == 'account-a'


@pytest.mark.parametrize('wiki', [False, True])
def test_feishu_folder_pages_and_full_listing_share_results(wiki):
    fs = (FeishuWikiFS if wiki else FeishuFS)(dynamic_auth=True, skip_instance_cache=True)
    field = 'items' if wiki else 'files'
    item_field = 'node_token' if wiki else 'token'

    def get(url, params):
        assert params['page_size'] == (50 if wiki else 200)
        assert params['parent_node_token' if wiki else 'folder_token'] == 'root'
        if wiki:
            assert url.endswith('/spaces/space/nodes')
        later = bool(params.get('page_token'))
        return {'code': 0, 'data': {
            field: [{item_field: 'b' if later else 'a'}],
            'has_more': not later, 'next_page_token': '' if later else 'next'}}
    fs._get = get
    kwargs = {'space_id': 'space'} if wiki else {'page_size': 200}
    first = fs.list_page('root', **kwargs)
    assert first == {'items': [{item_field: 'a'}], 'next_page_token': 'next'}
    full = fs._list_nodes_raw('root', 'space') if wiki else fs._list_files_raw('root')
    assert full == [{item_field: 'a'}, {item_field: 'b'}]


@pytest.mark.parametrize('wiki', [False, True])
@pytest.mark.parametrize('payload', [
    None, [], {'data': []},
    {'code': 403}, {'data': {'has_more': True}},
    {'data': {'has_more': True, 'page_token': 'same', 'items': [], 'files': []}},
    {'data': {'items': {}, 'files': {}}},
    {'data': {'items': [None], 'files': [None]}},
])
def test_feishu_rejects_incomplete_or_invalid_pages(wiki, payload):
    fs = (FeishuWikiFS if wiki else FeishuFS)(dynamic_auth=True, skip_instance_cache=True)
    fs._get = lambda *args, **kwargs: payload
    with pytest.raises(ValueError):
        fs.list_page('root', page_token='same', **({'space_id': 'space'} if wiki else {}))


@pytest.mark.parametrize('wiki', [False, True])
def test_feishu_empty_pages_continue_and_cursor_cycles_fail(wiki):
    fs = (FeishuWikiFS if wiki else FeishuFS)(dynamic_auth=True, skip_instance_cache=True)
    field = 'items' if wiki else 'files'
    pages = iter([
        {'data': {field: [], 'has_more': True, 'page_token': 'next'}},
        {'data': {field: [{'token': 'file'}], 'has_more': False}},
    ])
    fs._get = lambda *args, **kwargs: next(pages)
    kwargs = {'space_id': 'space'} if wiki else {}
    assert fs._all_folder_items('root', **kwargs) == [{'token': 'file'}]
    pages = iter({'data': {field: [], 'has_more': True, 'page_token': token}}
                 for token in ['a', 'b', 'a'])
    with pytest.raises(ValueError, match='repeated continuation token'):
        fs._all_folder_items('root', **kwargs)


@pytest.mark.parametrize('payload', [None, [], {'data': []}, {'data': {'folder': None}}])
def test_feishu_folder_metadata_rejects_invalid_response(payload):
    fs = FeishuFS(dynamic_auth=True, skip_instance_cache=True)
    fs._get = lambda *args, **kwargs: payload
    with pytest.raises(ValueError):
        fs.folder_info('folder')


@pytest.mark.parametrize('payload', [None, [], {'files': [None]}, {'files': {}}])
def test_google_folder_page_rejects_invalid_response(payload):
    fs = GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)
    fs._get = lambda *args, **kwargs: payload
    with pytest.raises(ValueError):
        fs.list_page(folder_id='folder')


@pytest.mark.parametrize('path, identifier', [
    ('/folder', 'folder'), ('/drive/shared', 'shared'),
    ('googledrive://drive/shared', 'shared'), ('/drive/shared/folder', 'folder'),
])
def test_google_folder_info_keeps_shared_drive_identity(path, identifier):
    fs = GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)

    def get(url, params):
        assert url.endswith('/files/' + identifier)
        assert 'driveId' in params['fields'].split(',')
        return {'id': identifier, 'name': 'Folder', 'mimeType': 'application/vnd.google-apps.folder',
                'driveId': 'shared'}
    fs._get = get
    info = fs.info(path)
    assert info['type'] == 'directory' and info['drive_id'] == 'shared'


@pytest.mark.parametrize('wrapped', [False, True])
def test_feishu_folder_metadata_uses_stable_id(wrapped):
    fs = FeishuFS(dynamic_auth=True, skip_instance_cache=True)
    info = {'token': 'folder', 'name': 'Reports'}

    def get(url):
        assert url.endswith('/drive/explorer/v2/folder/folder/meta')
        return {'code': 0, 'data': {'folder': info} if wrapped else info}

    fs._get = get
    assert fs.folder_info('folder') == info
