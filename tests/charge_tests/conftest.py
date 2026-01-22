import os
import re
import sys
import pytest
import lazyllm

def matches_any_pattern(changed_file, patterns):
    return any(re.fullmatch(pat, changed_file) for pat in patterns)

@pytest.fixture(autouse=True)
def ignore_cache_on_change(request):
    is_linux = sys.platform.startswith('linux')
    changed_files = [f.strip() for f in getattr(request.config, 'changed_files', []) if f and f.strip()]

    is_advanced_test = request.node.get_closest_marker('advanced_test') is not None
    is_connectivity_test = request.node.get_closest_marker('model_connectivity_test') is not None
    cache_marker = request.node.get_closest_marker('ignore_cache_on_change')

    force_cache = None
    if is_advanced_test:
        force_cache = True
    elif is_connectivity_test:
        force_cache = True
        if is_linux and cache_marker is not None:
            cared_files = cache_marker.args
            regex_mode = cache_marker.kwargs.get('regex_mode', False)
            if regex_mode:
                should_disable_cache = any(matches_any_pattern(f, cared_files) for f in changed_files)
            else:
                should_disable_cache = any(f in cared_files for f in changed_files)
            if should_disable_cache:
                force_cache = False
    elif cache_marker is not None:
        cared_files = cache_marker.args
        regex_mode = cache_marker.kwargs.get('regex_mode', False)
        if is_linux:
            if regex_mode:
                should_disable_cache = any(matches_any_pattern(f, cared_files) for f in changed_files)
            else:
                should_disable_cache = any(f in cared_files for f in changed_files)
            force_cache = not should_disable_cache
        else:
            force_cache = True

    old_value = os.environ.get('LAZYLLM_CACHE_ONLINE_MODULE')
    if force_cache is not None:
        os.environ['LAZYLLM_CACHE_ONLINE_MODULE'] = 'True' if force_cache else 'False'
        lazyllm.config.refresh('LAZYLLM_CACHE_ONLINE_MODULE')

    yield

    if old_value is None:
        os.environ.pop('LAZYLLM_CACHE_ONLINE_MODULE', None)
    else:
        os.environ['LAZYLLM_CACHE_ONLINE_MODULE'] = old_value
    if force_cache is not None:
        lazyllm.config.refresh('LAZYLLM_CACHE_ONLINE_MODULE')
