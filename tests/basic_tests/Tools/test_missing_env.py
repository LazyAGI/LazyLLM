import lazyllm
from lazyllm.tools.agent.missing_env import (
    collect_missing_env_hints,
    format_missing_env_message,
    normalize_required_env_names,
)
from lazyllm.tools.agent.toolError import ToolExecutionError, exception_failure


def _restore_dynamic_env(old_dynamic_env):
    if old_dynamic_env is None:
        lazyllm.globals.pop('dynamic_env_vars', None)
    else:
        lazyllm.globals['dynamic_env_vars'] = old_dynamic_env


def test_normalize_required_env_names_from_list_and_csv():
    assert normalize_required_env_names(['REDFOX_API_KEY', 'REDFOX_API_KEY', 'bad name']) == [
        'REDFOX_API_KEY',
    ]
    assert normalize_required_env_names('KEY_A, KEY_B') == ['KEY_A', 'KEY_B']
    assert normalize_required_env_names(None) == []
    assert normalize_required_env_names(True) == []


def test_collect_missing_env_hints_layers_convention_declared_and_heuristic(monkeypatch):
    old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
    lazyllm.globals['dynamic_env_vars'] = {}
    monkeypatch.delenv('DECLARED_API_KEY', raising=False)
    monkeypatch.delenv('HEURISTIC_API_KEY', raising=False)
    try:
        names = collect_missing_env_hints(
            'MISSING_ENV=CONVENTION_API_KEY\nmissing HEURISTIC_API_KEY\nmissing file',
            declared_required=['DECLARED_API_KEY', 'CONVENTION_API_KEY'],
        )
    finally:
        _restore_dynamic_env(old_dynamic_env)

    assert names == ['CONVENTION_API_KEY', 'DECLARED_API_KEY', 'HEURISTIC_API_KEY']


def test_collect_missing_env_hints_skips_set_declared_and_plain_words(monkeypatch):
    old_dynamic_env = lazyllm.globals.get('dynamic_env_vars')
    lazyllm.globals['dynamic_env_vars'] = {'DECLARED_API_KEY': 'already'}
    monkeypatch.delenv('API_KEY', raising=False)
    try:
        names = collect_missing_env_hints(
            'missing file\nAPI_KEY\nPATH is required',
            declared_required=['DECLARED_API_KEY'],
        )
    finally:
        _restore_dynamic_env(old_dynamic_env)

    assert names == []


def test_format_missing_env_message_and_exception_failure_keep_original_text():
    reason = 'Skill script execution failed with exit code 1: boom'
    message = format_missing_env_message(reason, ['REDFOX_API_KEY'])
    error = ToolExecutionError.with_missing_env(message, ['REDFOX_API_KEY'])
    failure = exception_failure('run_script', error)

    assert reason in message
    assert 'missing_env: ["REDFOX_API_KEY"]' in message
    assert failure['ok'] is False
    assert failure['missing_env'] == ['REDFOX_API_KEY']
    assert failure['value'] == message
