import os
import re
import sys
import pytest
import lazyllm

def matches_any_pattern(changed_file, patterns):
    return any(re.fullmatch(pat, changed_file) for pat in patterns)

def is_main_branch():
    return (os.environ.get('GIT_BRANCH') == 'main'
            or os.environ.get('CI_COMMIT_REF_NAME') == 'main'
            or os.environ.get('LAZYLLM_ON_MAIN') == 'True')

@pytest.fixture(autouse=True)
def ignore_cache_on_change(request):
    is_linux = sys.platform.startswith('linux')
    changed_files = [f.strip() for f in getattr(request.config, 'changed_files', []) if f and f.strip()]

    is_advanced_test = request.node.get_closest_marker('advanced_test') is not None
    is_connectivity_test = request.node.get_closest_marker('model_connectivity_test') is not None
    cache_marker = request.node.get_closest_marker('ignore_cache_on_change')

    is_rerun = getattr(request.node, 'execution_count', 1) > 1

    # Default behavior: enable cache
    force_cache = True
    should_ignore_cache = False

    if is_rerun:
        should_ignore_cache = True
    elif is_advanced_test:
        force_cache = True
        should_ignore_cache = False
    elif is_connectivity_test:
        force_cache = True
        if is_linux and cache_marker is not None:
            cared_files = cache_marker.args
            regex_mode = cache_marker.kwargs.get('regex_mode', False)
            should_ignore_cache = any(
                matches_any_pattern(f, cared_files) if regex_mode else f in cared_files
                for f in changed_files
            )
    elif cache_marker is not None:
        if is_linux:
            cared_files = cache_marker.args
            regex_mode = cache_marker.kwargs.get('regex_mode', False)
            should_ignore_cache = any(
                matches_any_pattern(f, cared_files) if regex_mode else f in cared_files
                for f in changed_files
            )
        else:
            force_cache = True
            should_ignore_cache = False

    cache_mode = 'RW'
    if should_ignore_cache:
        if is_main_branch():
            force_cache = True
            cache_mode = 'WO'
        else:
            force_cache = False
            cache_mode = 'NONE'

    old_cache_online = os.environ.get('LAZYLLM_CACHE_ONLINE_MODULE')
    old_cache_mode = os.environ.get('LAZYLLM_CACHE_MODE')

    os.environ['LAZYLLM_CACHE_ONLINE_MODULE'] = 'True' if force_cache else 'False'
    os.environ['LAZYLLM_CACHE_MODE'] = cache_mode
    lazyllm.config.refresh('LAZYLLM_CACHE_ONLINE_MODULE')
    lazyllm.config.refresh('LAZYLLM_CACHE_MODE')

    yield

    if old_cache_online is None:
        os.environ.pop('LAZYLLM_CACHE_ONLINE_MODULE', None)
    else:
        os.environ['LAZYLLM_CACHE_ONLINE_MODULE'] = old_cache_online

    if old_cache_mode is None:
        os.environ.pop('LAZYLLM_CACHE_MODE', None)
    else:
        os.environ['LAZYLLM_CACHE_MODE'] = old_cache_mode

    lazyllm.config.refresh('LAZYLLM_CACHE_ONLINE_MODULE')
    lazyllm.config.refresh('LAZYLLM_CACHE_MODE')
