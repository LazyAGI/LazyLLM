import os
import re
import pytest

def matches_any_pattern(changed_file, patterns):
    return any(re.fullmatch(pat, changed_file) for pat in patterns)

@pytest.fixture(autouse=True)
def ignore_cache_on_change(request):
    marker = request.node.get_closest_marker('ignore_cache_on_change')
    if marker is None:
        yield
        return

    cared_files = marker.args
    regex_mode = marker.kwargs.get('regex_mode', False)
    if regex_mode:
        ignore_cache = any(matches_any_pattern(f, cared_files) for f in request.config.changed_files)
    else:
        ignore_cache = any(f in cared_files for f in request.config.changed_files)

    old_value = os.environ.get('LAZYLLM_CACHE_ONLINE_MODULE')
    os.environ['LAZYLLM_CACHE_ONLINE_MODULE'] = 'False' if ignore_cache else 'True'

    yield

    if old_value is None:
        os.environ.pop('LAZYLLM_CACHE_ONLINE_MODULE', None)
    else:
        os.environ['LAZYLLM_CACHE_ONLINE_MODULE'] = old_value
