import os
import re
import pytest

def pytest_configure(config):
    env_str = os.getenv('CHANGED_FILES')  # Set outside pytest.
    config.changed_files = env_str.split(',') if env_str is not None else []
    config.disable_run_on_change = os.getenv('DISABLE_RUN_ON_CHANGE', '')\
        .lower() in ('1', 'true', 'yes', 'on')

def matches_any_pattern(changed_file, patterns):
    return any(re.fullmatch(pat, changed_file) for pat in patterns)

def pytest_runtest_setup(item):
    if (marker := item.get_closest_marker('run_on_change')) is not None:
        if item.config.disable_run_on_change:
            return

        files = marker.args
        regex_mode = marker.kwargs.get('regex_mode', False)

        if regex_mode:
            if not any(matches_any_pattern(f, files) for f in item.config.changed_files):
                pytest.skip(f'Skipped: none of "{files}" matched the changed files.')
        else:
            if not any(f in files for f in item.config.changed_files):
                pytest.skip(f'Skipped: none of "{files}" were changed.')
