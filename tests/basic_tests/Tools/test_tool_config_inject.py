import os

import lazyllm
from lazyllm.tools.tool_config_inject import (
    TOOL_AUTH_REGISTRY,
    effective_env_value,
    get_dynamic_env_vars,
    inject_env_vars,
    inject_tool_config,
    register_tool_auth,
)


def _restore_dynamic_env(old_dynamic_env):
    if old_dynamic_env is None:
        lazyllm.globals.pop('dynamic_env_vars', None)
    else:
        lazyllm.globals['dynamic_env_vars'] = old_dynamic_env


def test_inject_env_vars_overwrites_and_clears_by_empty_string():
    old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
    lazyllm.globals['dynamic_env_vars'] = {}
    try:
        inject_env_vars({'REDFOX_API_KEY': 'first', 'KEEP_TOKEN': 'keep'})
        inject_env_vars({'REDFOX_API_KEY': 'second'})
        assert get_dynamic_env_vars()['REDFOX_API_KEY'] == 'second'
        inject_env_vars({'REDFOX_API_KEY': '  '})
        assert 'REDFOX_API_KEY' not in get_dynamic_env_vars()
        assert get_dynamic_env_vars()['KEEP_TOKEN'] == 'keep'
    finally:
        _restore_dynamic_env(old_dynamic_env)


def test_inject_env_vars_skips_invalid_name_and_nul_value():
    old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
    lazyllm.globals['dynamic_env_vars'] = {}
    try:
        inject_env_vars({
            'RED FOX': 'nope',
            'GOOD_KEY': 'ok',
            'NUL_KEY': 'abc\0def',
            '\0BAD': 'x',
            'SKIP_NONE': None,
        })
        assert get_dynamic_env_vars() == {'GOOD_KEY': 'ok'}
    finally:
        _restore_dynamic_env(old_dynamic_env)


def test_effective_env_value_prefers_dynamic_over_os_environ(monkeypatch):
    old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
    lazyllm.globals['dynamic_env_vars'] = {}
    monkeypatch.delenv('SESSION_ONLY_KEY', raising=False)
    try:
        assert effective_env_value('SESSION_ONLY_KEY') == ''
        inject_env_vars({'SESSION_ONLY_KEY': 'from-session'})
        assert os.getenv('SESSION_ONLY_KEY') is None
        assert effective_env_value('SESSION_ONLY_KEY') == 'from-session'
    finally:
        _restore_dynamic_env(old_dynamic_env)


_MAIL_PROVIDER_NAMES = {
    'gmailimap', 'qqmail', 'qqexmail', 'netease163', 'neteaseqiye',
}


def test_builtin_registry_has_no_mail_provider_names():
    assert _MAIL_PROVIDER_NAMES.isdisjoint(TOOL_AUTH_REGISTRY)


def test_register_tool_auth_maps_external_tool():
    lazyllm.globals._init_sid(sid='test-register-tool-auth')
    previous = TOOL_AUTH_REGISTRY.get('custom_auth_tool')
    try:
        register_tool_auth('custom_auth_tool', 'dynamic_tool_auth')
        assert TOOL_AUTH_REGISTRY['custom_auth_tool'] == 'dynamic_tool_auth'
        inject_tool_config({'custom_auth_tool': 'tok-1'})
        assert lazyllm.globals.config['dynamic_tool_auth']['custom_auth_tool'] == 'tok-1'
    finally:
        if previous is None:
            TOOL_AUTH_REGISTRY.pop('custom_auth_tool', None)
        else:
            TOOL_AUTH_REGISTRY['custom_auth_tool'] = previous
        lazyllm.globals.clear()


def test_register_tool_auth_rejects_unknown_bucket():
    try:
        register_tool_auth('custom_auth_tool', 'not_a_bucket')
        raise AssertionError('expected ValueError')
    except ValueError as orig:
        assert 'config_key' in str(orig)


def test_register_tool_auth_conflict_policy():
    previous = TOOL_AUTH_REGISTRY.get('conflict_auth_tool')
    try:
        register_tool_auth('conflict_auth_tool', 'dynamic_tool_auth')
        register_tool_auth('conflict_auth_tool', 'dynamic_tool_auth')
        try:
            register_tool_auth('conflict_auth_tool', 'dynamic_fs_auth')
            raise AssertionError('expected ValueError')
        except ValueError as orig:
            assert 'already registered' in str(orig)
        assert TOOL_AUTH_REGISTRY['conflict_auth_tool'] == 'dynamic_tool_auth'

        register_tool_auth(
            'conflict_auth_tool', 'dynamic_fs_auth', on_conflict='ignore',
        )
        assert TOOL_AUTH_REGISTRY['conflict_auth_tool'] == 'dynamic_tool_auth'

        register_tool_auth(
            'conflict_auth_tool', 'dynamic_fs_auth', on_conflict='replace',
        )
        assert TOOL_AUTH_REGISTRY['conflict_auth_tool'] == 'dynamic_fs_auth'
    finally:
        if previous is None:
            TOOL_AUTH_REGISTRY.pop('conflict_auth_tool', None)
        else:
            TOOL_AUTH_REGISTRY['conflict_auth_tool'] = previous


def test_github_tool_config_is_injected_as_dynamic_fs_auth():
    old = lazyllm.globals.config['dynamic_fs_auth']
    lazyllm.globals.config['dynamic_fs_auth'] = {}
    try:
        inject_tool_config({'github': 'github-token'})
        assert lazyllm.globals.config['dynamic_fs_auth']['github'] == 'github-token'
    finally:
        lazyllm.globals.config['dynamic_fs_auth'] = old
