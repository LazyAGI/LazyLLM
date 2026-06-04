import threading
import pytest
from unittest.mock import patch

from lazyllm.common import KeyPool, KeySelectPolicy, KeyAuthError, AllKeysExhaustedError, Credential, CredentialMixin
from lazyllm.common.auth import BearerTokenStrategy
from lazyllm.common.globals import globals as lazyllm_globals, locals as lazyllm_locals


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_globals():
    lazyllm_globals['key_pool_state'] = {}


class _MockService(CredentialMixin):
    def __init__(self, tokens, policy=KeySelectPolicy.RANDOM, dynamic_auth=False, skip_auth=False):
        cred = self._default_credential(tokens, dynamic_auth=dynamic_auth)
        self.__init_credential__(cred, strategy=BearerTokenStrategy(),
                                 skip_auth=skip_auth, dynamic_key_policy=policy)

    def _is_key_auth_error(self, resp):
        return getattr(resp, 'status_code', 0) in (401, 403)


class _DynamicService(CredentialMixin):
    def __init__(self, token_source, policy=KeySelectPolicy.RANDOM):
        cred = self._default_credential(None, dynamic_auth=True)
        self.__init_credential__(cred, strategy=BearerTokenStrategy(), dynamic_key_policy=policy)
        self._token_source = token_source

    def _resolve_dynamic_token(self):
        return self._token_source


# ---------------------------------------------------------------------------
# KeySelectPolicy & KeyPool unit tests
# ---------------------------------------------------------------------------

class TestKeyPool:

    def setup_method(self):
        _fresh_globals()

    def test_single_key_ordered_random(self):
        pool = KeyPool(['k1'], KeySelectPolicy.RANDOM)
        assert pool.ordered_keys() == ['k1']

    def test_random_excludes_failed_key(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.RANDOM)
        pool.report_failure('k1')
        keys = pool.ordered_keys()
        assert 'k1' not in keys
        assert set(keys) == {'k2', 'k3'}

    def test_round_robin_order(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.ROUND_ROBIN)
        r1 = pool.ordered_keys()
        r2 = pool.ordered_keys()
        r3 = pool.ordered_keys()
        r4 = pool.ordered_keys()
        assert r1[0] == 'k1'
        assert r2[0] == 'k2'
        assert r3[0] == 'k3'
        assert r4[0] == 'k1'  # wraps around

    def test_round_robin_excludes_failed(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.ROUND_ROBIN)
        pool.report_failure('k1')
        keys = pool.ordered_keys()
        assert 'k1' not in keys

    def test_round_robin_empty_after_all_failed(self):
        pool = KeyPool(['k1', 'k2'], KeySelectPolicy.ROUND_ROBIN)
        pool.report_failure('k1')
        pool.report_failure('k2')
        assert pool.ordered_keys() == []

    def test_prefer_last_success_order(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.PREFER_LAST_SUCCESS)
        pool.report_success('k2')
        keys = pool.ordered_keys()
        assert keys[0] == 'k2'

    def test_prefer_last_success_failed_at_end_not_removed(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.PREFER_LAST_SUCCESS)
        pool.report_failure('k1')
        keys = pool.ordered_keys()
        assert 'k1' in keys
        assert keys[-1] == 'k1'

    def test_report_success_updates_last_success(self):
        pool = KeyPool(['k1', 'k2'], KeySelectPolicy.PREFER_LAST_SUCCESS)
        pool.report_success('k1')
        assert pool._get_state()['last_success'] == 'k1'
        pool.report_success('k2')
        assert pool._get_state()['last_success'] == 'k2'

    def test_report_success_does_not_clear_failed(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.RANDOM)
        pool.report_failure('k1')
        pool.report_success('k2')
        assert 'k1' in pool._get_state().get('failed', set())

    def test_peek_random_returns_first_non_failed(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.RANDOM)
        pool.report_failure('k1')
        result = pool.peek()
        assert result != 'k1'
        assert result in ('k2', 'k3')

    def test_peek_prefer_last_success_returns_last_success(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.PREFER_LAST_SUCCESS)
        pool.report_success('k3')
        assert pool.peek() == 'k3'

    def test_peek_no_side_effect_on_rr_index(self):
        pool = KeyPool(['k1', 'k2', 'k3'], KeySelectPolicy.ROUND_ROBIN)
        _ = pool.peek()
        _ = pool.peek()
        # rr_index should NOT have been advanced by peek
        assert pool.ordered_keys()[0] == 'k1'

    def test_peek_empty_returns_empty_string(self):
        pool = KeyPool(['k1'], KeySelectPolicy.RANDOM)
        pool.report_failure('k1')
        assert pool.peek() == ''


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------

class TestCredential:

    def test_key_pool_only_allowed_for_static(self):
        pool = KeyPool(['k1', 'k2'], KeySelectPolicy.RANDOM)
        with pytest.raises(ValueError, match='key_pool only allowed'):
            Credential(kind='oauth2', key_pool=pool)

    def test_key_pool_allowed_for_static(self):
        pool = KeyPool(['k1', 'k2'], KeySelectPolicy.RANDOM)
        cred = Credential(kind='static', key_pool=pool)
        assert cred.key_pool is pool

    def test_key_pool_allowed_for_dynamic(self):
        pool = KeyPool(['k1', 'k2'], KeySelectPolicy.RANDOM)
        cred = Credential(kind='dynamic', key_pool=pool)
        assert cred.key_pool is pool

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match='Invalid Credential.kind'):
            Credential(kind='unknown')


# ---------------------------------------------------------------------------
# CredentialMixin._default_credential
# ---------------------------------------------------------------------------

class TestDefaultCredential:

    def test_single_token_builds_static(self):
        svc = _MockService('mykey')
        assert svc._credential.kind == 'static'
        assert svc._credential.key_pool is None

    def test_list_with_one_element_builds_static_no_pool(self):
        svc = _MockService(['onlykey'])
        assert svc._credential.kind == 'static'
        assert svc._credential.key_pool is None
        assert svc._credential.secret_key == 'onlykey'

    def test_list_with_multiple_elements_builds_key_pool(self):
        svc = _MockService(['k1', 'k2', 'k3'])
        assert svc._credential.key_pool is not None
        assert set(svc._credential.key_pool._keys) == {'k1', 'k2', 'k3'}

    def test_dynamic_auth_builds_dynamic_credential(self):
        svc = _MockService(None, dynamic_auth=True)
        assert svc._credential.kind == 'dynamic'


# ---------------------------------------------------------------------------
# CredentialMixin._request – multi-key rotation
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, ok=True):
        self.status_code = status_code
        self.ok = ok


class TestRequestMultiKey:

    def setup_method(self):
        _fresh_globals()

    def _make_svc(self, keys, policy=KeySelectPolicy.ROUND_ROBIN):
        svc = _MockService(keys, policy=policy)
        return svc

    def test_single_key_path_no_pool(self):
        svc = _MockService('only_key')
        with patch.object(svc, '_http_execute', return_value=_FakeResp(200)) as mock_exec:
            svc._request('GET', 'http://example.com')
        mock_exec.assert_called_once()
        _, call_kwargs = mock_exec.call_args
        assert 'Bearer only_key' in call_kwargs.get('headers', {}).get('Authorization', '')

    def test_multi_key_uses_correct_key_in_header(self):
        svc = self._make_svc(['k1', 'k2', 'k3'], KeySelectPolicy.ROUND_ROBIN)
        captured_headers = []

        def fake_execute(method, url, **kwargs):
            captured_headers.append(kwargs.get('headers', {}))
            return _FakeResp(200)

        # make two consecutive requests and check that they use different keys (rotation)
        with patch.object(svc, '_http_execute', side_effect=fake_execute):
            svc._request('GET', 'http://example.com')
            svc._request('GET', 'http://example.com')
        auth0 = captured_headers[0].get('Authorization', '')
        auth1 = captured_headers[1].get('Authorization', '')
        assert auth0.startswith('Bearer ')
        assert auth1.startswith('Bearer ')
        assert auth0 != auth1, 'ROUND_ROBIN should use different keys on consecutive requests'

    def test_multi_key_rotates_on_auth_error(self):
        # Use PREFER_LAST_SUCCESS so the first key is deterministic (last_success starts empty → k1 first)
        svc = self._make_svc(['k1', 'k2', 'k3'], KeySelectPolicy.PREFER_LAST_SUCCESS)
        call_count = [0]

        def fake_execute(method, url, **kwargs):
            call_count[0] += 1
            _ = kwargs.get('headers', {}).get('Authorization', '')
            # Fail the first attempted key
            if call_count[0] == 1:
                raise KeyAuthError('first key invalid')
            return _FakeResp(200)

        with patch.object(svc, '_http_execute', side_effect=fake_execute):
            svc._request('GET', 'http://example.com')
        assert call_count[0] == 2  # first key failed, second key succeeded

    def test_all_keys_exhausted_raises(self):
        svc = self._make_svc(['k1', 'k2'], KeySelectPolicy.ROUND_ROBIN)

        def always_auth_fail(method, url, **kwargs):
            raise KeyAuthError('always fail')

        with patch.object(svc, '_http_execute', side_effect=always_auth_fail):
            with pytest.raises(AllKeysExhaustedError):
                svc._request('GET', 'http://example.com')

    def test_failed_keys_marked_in_pool(self):
        svc = self._make_svc(['k1', 'k2', 'k3'], KeySelectPolicy.PREFER_LAST_SUCCESS)
        first_key = [None]

        def fail_first(method, url, **kwargs):
            auth = kwargs.get('headers', {}).get('Authorization', '')
            key = auth.replace('Bearer ', '')
            if first_key[0] is None:
                first_key[0] = key
                raise KeyAuthError(key)
            return _FakeResp(200)

        with patch.object(svc, '_http_execute', side_effect=fail_first):
            svc._request('GET', 'http://example.com')

        pool = svc._credential.key_pool
        failed = pool._get_state().get('failed', set())
        assert first_key[0] is not None
        assert first_key[0] in failed

    def test_success_reported_in_pool(self):
        svc = self._make_svc(['k1', 'k2'], KeySelectPolicy.ROUND_ROBIN)

        with patch.object(svc, '_http_execute', return_value=_FakeResp(200)):
            svc._request('GET', 'http://example.com')

        pool = svc._credential.key_pool
        last = pool._get_state().get('last_success')
        assert last in ('k1', 'k2')

    def test_non_auth_error_propagates_without_key_rotation(self):
        import requests as _requests
        svc = self._make_svc(['k1', 'k2'], KeySelectPolicy.ROUND_ROBIN)
        call_count = [0]

        def network_error(method, url, **kwargs):
            call_count[0] += 1
            raise _requests.ConnectionError('network')

        with patch.object(svc, '_http_execute', side_effect=network_error):
            with pytest.raises(_requests.ConnectionError):
                svc._request('GET', 'http://example.com')
        assert call_count[0] == 1  # no retry

    def test_curr_key_cleared_after_request(self):
        svc = self._make_svc(['k1', 'k2'], KeySelectPolicy.ROUND_ROBIN)
        with patch.object(svc, '_http_execute', return_value=_FakeResp(200)):
            svc._request('GET', 'http://example.com')
        assert svc._credential_id not in lazyllm_locals['curr_key']

    def test_curr_key_cleared_after_auth_error(self):
        svc = self._make_svc(['k1', 'k2'], KeySelectPolicy.ROUND_ROBIN)

        def fail(method, url, **kwargs):
            raise KeyAuthError('fail')

        with patch.object(svc, '_http_execute', side_effect=fail):
            with pytest.raises(AllKeysExhaustedError):
                svc._request('GET', 'http://example.com')
        assert svc._credential_id not in lazyllm_locals['curr_key']


# ---------------------------------------------------------------------------
# Dynamic multi-key
# ---------------------------------------------------------------------------

class TestDynamicMultiKey:

    def setup_method(self):
        _fresh_globals()
        lazyllm_locals['curr_key'] = {}

    def test_dynamic_list_builds_pool_in_globals(self):
        svc = _DynamicService(['k1', 'k2', 'k3'], policy=KeySelectPolicy.ROUND_ROBIN)
        pool = svc._get_active_pool()
        assert pool is not None
        assert set(pool._keys) == {'k1', 'k2', 'k3'}

    def test_dynamic_single_token_no_pool(self):
        svc = _DynamicService('only_key')
        pool = svc._get_active_pool()
        assert pool is None

    def test_dynamic_pool_not_stored_on_self(self):
        svc = _DynamicService(['k1', 'k2'], policy=KeySelectPolicy.ROUND_ROBIN)
        svc._get_active_pool()
        assert not hasattr(svc, '_dynamic_key_pool')

    def test_dynamic_pool_request_rotates_key(self):
        svc = _DynamicService(['k1', 'k2', 'k3'], policy=KeySelectPolicy.ROUND_ROBIN)
        call_count = [0]

        def fail_k1(method, url, **kwargs):
            call_count[0] += 1
            auth = kwargs.get('headers', {}).get('Authorization', '')
            if 'k1' in auth:
                raise KeyAuthError('k1')
            return _FakeResp(200)

        with patch.object(svc, '_http_execute', side_effect=fail_k1):
            svc._request('GET', 'http://example.com')
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Concurrent isolation: curr_key in Locals
# ---------------------------------------------------------------------------

class TestConcurrentIsolation:

    def setup_method(self):
        _fresh_globals()

    def test_concurrent_requests_use_independent_curr_key(self):
        svc = _MockService(['k1', 'k2', 'k3'], policy=KeySelectPolicy.ROUND_ROBIN)
        observed = {}
        barrier = threading.Barrier(3)

        def worker(tid, key_index):
            barrier.wait()
            # Directly manipulate curr_key as _request would
            lazyllm_locals['curr_key'][svc._credential_id] = f'key{key_index}'
            observed[tid] = lazyllm_locals['curr_key'].get(svc._credential_id)
            lazyllm_locals['curr_key'].pop(svc._credential_id, None)

        threads = [threading.Thread(target=worker, args=(i, i)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Each thread's observed key should match what it set (locals are thread-local)
        for tid, key in observed.items():
            assert key == f'key{tid}'


# ---------------------------------------------------------------------------
# skip_auth
# ---------------------------------------------------------------------------

class TestSkipAuth:

    def test_skip_auth_key_source_returns_true(self):
        svc = _MockService('key', skip_auth=True)
        assert svc.__key_source__() is True

    def test_no_skip_auth_key_source_returns_bool_of_token(self):
        svc = _MockService('mykey', skip_auth=False)
        assert svc.__key_source__() is True

    def test_no_skip_auth_empty_token_key_source_false(self):
        svc = _MockService('', skip_auth=False)
        assert svc.__key_source__() is False
