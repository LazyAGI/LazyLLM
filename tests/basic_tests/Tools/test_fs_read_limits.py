import io
from unittest.mock import Mock

import pytest
import requests

from lazyllm.tools.fs import FSReadLimitError, fs_read_limits
from lazyllm.tools.fs.limits import _bounded_request
from lazyllm.tools.fs.supplier.googledrive import GoogleDriveFS


def response(content=b'hello', status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result.raw = io.BytesIO(content)
    return result


def test_limits_stream_count_close_and_restore_legacy_behavior():
    session = Mock()
    first = response()
    first.close = Mock(wraps=first.close)
    session.request.return_value = first
    with fs_read_limits(max_bytes=6):
        assert _bounded_request(session, 'GET', 'https://test').content == b'hello'
        first.close.assert_called_once()
        assert session.request.call_args.kwargs['stream'] is True
        assert session.request.call_args.kwargs['allow_redirects'] is False
        assert session.request.call_args.kwargs['timeout'] == (5, 5)
        second = response(b'xx')
        session.request.return_value = second
        with pytest.raises(FSReadLimitError):
            _bounded_request(session, 'GET', 'https://test')
        assert second.raw.closed
    session.request.reset_mock()
    _bounded_request(session, 'POST', 'https://test', json={'existing': True})
    session.request.assert_called_once_with('POST', 'https://test', json={'existing': True})


@pytest.mark.parametrize(('status', 'headers'), [(200, {'Content-Length': '999'}), (302, {})])
def test_rejects_before_downloading_large_or_redirected_body(status, headers):
    result = response(status=status, headers=headers)
    result.iter_content = Mock(side_effect=AssertionError('must not download'))
    with fs_read_limits(max_bytes=10), pytest.raises(FSReadLimitError):
        _bounded_request(Mock(request=Mock(return_value=result)), 'GET', 'https://test')
    assert result.raw.closed


def test_deadline_request_count_and_nested_context(monkeypatch):
    from lazyllm.tools.fs import limits
    clock = [0]
    monkeypatch.setattr(limits.time, 'monotonic', lambda: clock[0])
    session = Mock(request=Mock(side_effect=lambda *args, **kwargs: response()))
    with fs_read_limits(seconds=1, max_requests=1):
        _bounded_request(session, 'GET', 'https://test')
        with fs_read_limits():
            _bounded_request(session, 'GET', 'https://test')
        with pytest.raises(FSReadLimitError):
            _bounded_request(session, 'GET', 'https://test')
        clock[0] = 2
        with pytest.raises(TimeoutError):
            _bounded_request(session, 'GET', 'https://test')
    assert session.request.call_count == 2


def test_single_page_preserves_empty_continuation_scope_and_warning():
    fs = GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)
    fs._get = Mock(return_value={'files': [], 'nextPageToken': 'continue', 'incompleteSearch': True})
    page = fs.list_page(folder_id='folder-1', drive_id='drive-1', query="owner's", page_size=3)
    assert page == {'items': [], 'next_page_token': 'continue', 'incomplete_search': True}
    query = fs._get.call_args.kwargs['params']
    assert query['q'] == "trashed = false and 'folder-1' in parents and name contains 'owner\\'s'"
    assert query['pageSize'] == 3 and query['driveId'] == 'drive-1'
    fs._get.assert_called_once()
    fs.list_page(query='term', query_mode='full_text', page_token=page['next_page_token'])
    query = fs._get.call_args.kwargs['params']
    assert query['pageToken'] == 'continue'
    assert "fullText contains 'term'" in query['q']


@pytest.mark.parametrize('kwargs', [
    {'page_size': 0}, {'page_size': True}, {'folder_id': "x' or true"},
    {'drive_id': '/etc'}, {'query_mode': 'raw_sql'},
])
def test_single_page_validates_before_io(kwargs):
    fs = GoogleDriveFS(dynamic_auth=True, skip_instance_cache=True)
    fs._get = Mock()
    with pytest.raises(ValueError):
        fs.list_page(**kwargs)
    fs._get.assert_not_called()
