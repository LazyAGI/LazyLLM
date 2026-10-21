import subprocess

import pytest

from lazyllm.tools.git.review.utils import _parse_unified_diff


@pytest.mark.parametrize('before, after, expected', [
    ('old\n', 'new\n', [(1, 1)]),
    ('', 'new\n', [(1, 1)]),
    ('old\nkeep\nlast\n', 'new\nkeep\nchanged\n', [(1, 1), (3, 1)]),
    ('old\n', 'first\nsecond\n', [(1, 2)]),
    ('old\n', '', []),
])
def test_real_git_zero_context_hunks(tmp_path, before, after, expected):
    old, new = tmp_path / 'before.txt', tmp_path / 'after.txt'
    old.write_text(before)
    new.write_text(after)
    process = subprocess.run([
        'git', 'diff', '--no-index', '--unified=0', str(old), str(new),
    ], capture_output=True, text=True)
    assert process.returncode == 1
    hunks = _parse_unified_diff(process.stdout)
    assert [(start, count) for _, start, count, _ in hunks] == expected
    for path, _, _, content in hunks:
        assert path.endswith('after.txt')
        assert '+' in content


def test_single_line_hunk_after_explicit_count_does_not_reuse_count():
    diff = '\n'.join([
        'diff --git a/example.py b/example.py',
        '@@ -1,2 +1,2 @@', '-a', '-b', '+c', '+d',
        '@@ -10 +10 @@ function +123,45', '-old', '+new',
    ])
    assert [(start, count) for _, start, count, _ in _parse_unified_diff(diff)] == [(1, 2), (10, 1)]
