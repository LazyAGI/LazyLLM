import os

import lazyllm
from lazyllm.tools.tool_config_inject import (
    effective_env_value,
    get_dynamic_env_vars,
    inject_env_vars,
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
