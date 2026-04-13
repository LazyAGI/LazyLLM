# Tests for diff line-number annotation and large-hunk splitting logic.
import textwrap
import pytest

from lazyllm.tools.git.review.utils import _annotate_diff_with_line_numbers, _parse_unified_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_annotated(annotated: str):
    '''Return list of (old, new, raw_line) from annotated output.
    old/new are int or None (when '--').
    '''
    result = []
    for line in annotated.splitlines():
        # prefix format: [NNN|NNN] or [--|NNN] or [NNN|--]
        assert line.startswith('['), f'unexpected line: {line!r}'
        bracket_end = line.index(']')
        parts = line[1:bracket_end].split('|')
        assert len(parts) == 2, f'bad prefix: {line!r}'
        old_s, new_s = parts
        old = None if old_s.strip() == '--' else int(old_s)
        new = None if new_s.strip() == '--' else int(new_s)
        raw = line[bracket_end + 2:]  # skip '] '
        result.append((old, new, raw))
    return result


# ---------------------------------------------------------------------------
# _annotate_diff_with_line_numbers
# ---------------------------------------------------------------------------

class TestAnnotateDiffWithLineNumbers:
    def test_pure_context(self):
        content = ' line1\n line2\n line3'
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=10))
        assert rows == [
            (10, 10, ' line1'),
            (11, 11, ' line2'),
            (12, 12, ' line3'),
        ]

    def test_pure_additions(self):
        content = '+added1\n+added2'
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=5))
        assert rows == [
            (None, 5, '+added1'),
            (None, 6, '+added2'),
        ]

    def test_pure_deletions(self):
        content = '-del1\n-del2'
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=3))
        assert rows == [
            (3, None, '-del1'),
            (4, None, '-del2'),
        ]

    def test_mixed_hunk(self):
        # Simulates: @@ -16,8 +16,10 @@
        # old: 16..23 (8 lines), new: 16..25 (10 lines)
        content = textwrap.dedent('''\
             R1 = 'r1'
             R2 = 'r2'
             R3 = 'r3'
            +R4A = 'r4a'
             R4 = 'r4'
             FINAL = 'final'
            +UPLOAD = 'upload'
             \n''').rstrip('\n')
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=16))
        # context lines advance both counters
        assert rows[0] == (16, 16, " R1 = 'r1'")
        assert rows[1] == (17, 17, " R2 = 'r2'")
        assert rows[2] == (18, 18, " R3 = 'r3'")
        # added line: old stays, new advances
        assert rows[3] == (None, 19, "+R4A = 'r4a'")
        assert rows[4] == (19, 20, " R4 = 'r4'")
        assert rows[5] == (20, 21, " FINAL = 'final'")
        assert rows[6] == (None, 22, "+UPLOAD = 'upload'")

    def test_deletion_then_addition(self):
        # Replace one line with two lines
        content = '-old_line\n+new_line_a\n+new_line_b\n context'
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=1))
        assert rows[0] == (1, None, '-old_line')
        assert rows[1] == (None, 1, '+new_line_a')
        assert rows[2] == (None, 2, '+new_line_b')
        assert rows[3] == (2, 3, ' context')

    def test_new_file_numbers_are_contiguous(self):
        # new-file line numbers must be strictly increasing with no gaps
        content = ' ctx\n+add1\n-del1\n+add2\n ctx2'
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=100))
        new_nos = [r[1] for r in rows if r[1] is not None]
        assert new_nos == list(range(100, 100 + len(new_nos)))

    def test_start_at_one(self):
        content = '+first\n ctx\n-gone'
        rows = _parse_annotated(_annotate_diff_with_line_numbers(content, new_start=1))
        assert rows[0][1] == 1   # added → new=1
        assert rows[1][1] == 2   # context → new=2
        assert rows[2][1] is None  # deleted → no new

    def test_empty_content(self):
        assert _annotate_diff_with_line_numbers('', new_start=1) == ''


# ---------------------------------------------------------------------------
# Large-hunk window splitting: win_start computation
# ---------------------------------------------------------------------------

def _compute_win_starts(hunk_lines, new_start, window_lines, overlap=30):
    '''Replicate the win_start logic from _analyze_large_hunk without calling LLM.'''
    step = max(1, window_lines - overlap)
    total = len(hunk_lines)
    results = []
    win_idx = 0
    while win_idx * step < total:
        start_offset = win_idx * step
        end_offset = min(start_offset + window_lines, total)
        win_lines = hunk_lines[start_offset:end_offset]
        new_file_before = sum(1 for ln in hunk_lines[:start_offset] if not ln.startswith('-'))
        win_start = new_start + new_file_before
        win_count = sum(1 for ln in win_lines if not ln.startswith('-'))
        results.append((win_start, win_count, win_lines))
        win_idx += 1
    return results


class TestLargeHunkSplitting:
    def test_all_context_lines(self):
        # 10 context lines, window=4, overlap=1 → step=3
        lines = [f' line{i}\n' for i in range(1, 11)]
        wins = _compute_win_starts(lines, new_start=1, window_lines=4, overlap=1)
        # windows: [0:4], [3:7], [6:10], [9:10]
        assert wins[0][0] == 1   # win_start = 1 + 0 context before
        assert wins[1][0] == 4   # 3 context lines before → 1+3=4
        assert wins[2][0] == 7   # 6 context lines before → 1+6=7
        assert wins[3][0] == 10  # 9 context lines before → 1+9=10

    def test_deletions_dont_advance_new_start(self):
        # 5 deleted lines then 5 context lines, window=4, overlap=1
        lines = [f'-del{i}\n' for i in range(5)] + [f' ctx{i}\n' for i in range(5)]
        wins = _compute_win_starts(lines, new_start=10, window_lines=4, overlap=1)
        # window 0: lines[0:4] = 4 deletions → win_start=10, win_count=0
        assert wins[0][0] == 10
        assert wins[0][1] == 0
        # window 1: lines[3:7] = 1 del + 3 ctx → new_file_before = 0 (all dels) → win_start=10
        assert wins[1][0] == 10
        # window 2: lines[6:10] = 4 ctx → new_file_before = 1 del (lines[0:6] has 1 ctx at idx5)
        # lines[0:6]: 5 dels + 1 ctx → 1 non-del → win_start = 10+1 = 11
        assert wins[2][0] == 11

    def test_additions_advance_new_start(self):
        # 5 added lines then 5 context lines, window=4, overlap=1
        lines = [f'+add{i}\n' for i in range(5)] + [f' ctx{i}\n' for i in range(5)]
        wins = _compute_win_starts(lines, new_start=1, window_lines=4, overlap=1)
        # window 0: lines[0:4] = 4 adds → win_start=1, win_count=4
        assert wins[0][0] == 1
        assert wins[0][1] == 4
        # window 1: lines[3:7] = 2 adds + 2 ctx → new_file_before = 3 adds → win_start=4
        assert wins[1][0] == 4
        # window 2: lines[6:10] = 4 ctx → new_file_before = 5 adds + 1 ctx = 6 → win_start=7
        assert wins[2][0] == 7

    def test_win_start_matches_annotated_first_new_line(self):
        # The first new-file line number in the annotated window must equal win_start.
        lines = ['+add\n', ' ctx\n', '-del\n', '+add2\n', ' ctx2\n',
                 ' ctx3\n', '+add3\n', ' ctx4\n']
        wins = _compute_win_starts(lines, new_start=10, window_lines=4, overlap=1)
        for win_start, win_count, win_lines in wins:
            win_content = ''.join(win_lines).rstrip('\n')
            annotated = _annotate_diff_with_line_numbers(win_content, win_start)
            rows = _parse_annotated(annotated)
            # first new-file line number in this window
            first_new = next((r[1] for r in rows if r[1] is not None), None)
            if first_new is not None:
                assert first_new == win_start, (
                    f'win_start={win_start} but first annotated new-line={first_new}'
                )

    def test_no_overlap_between_windows_new_lines(self):
        # With overlap=0, each window's new-file range should be disjoint.
        lines = [f' ctx{i}\n' for i in range(9)]
        wins = _compute_win_starts(lines, new_start=1, window_lines=3, overlap=0)
        ranges = [(ws, ws + wc) for ws, wc, _ in wins]
        for i in range(len(ranges) - 1):
            assert ranges[i][1] <= ranges[i + 1][0], (
                f'overlapping new-file ranges: {ranges[i]} and {ranges[i+1]}'
            )


# ---------------------------------------------------------------------------
# _parse_unified_diff: verify new_start / new_count extraction
# ---------------------------------------------------------------------------

class TestParseUnifiedDiff:
    def test_basic_hunk(self):
        diff = textwrap.dedent('''\
            diff --git a/foo.py b/foo.py
            index abc..def 100644
            --- a/foo.py
            +++ b/foo.py
            @@ -16,8 +16,10 @@
             R1 = 'r1'
            +R4A = 'r4a'
             R4 = 'r4'
            ''')
        hunks = _parse_unified_diff(diff)
        assert len(hunks) == 1
        path, new_start, new_count, content = hunks[0]
        assert path == 'foo.py'
        assert new_start == 16
        assert new_count == 10

    def test_multiple_hunks_same_file(self):
        diff = textwrap.dedent('''\
            diff --git a/bar.py b/bar.py
            --- a/bar.py
            +++ b/bar.py
            @@ -1,3 +1,4 @@
             a
            +b
             c
             d
            @@ -10,2 +11,3 @@
             x
            +y
             z
            ''')
        hunks = _parse_unified_diff(diff)
        assert len(hunks) == 2
        assert hunks[0][1] == 1 and hunks[0][2] == 4
        assert hunks[1][1] == 11 and hunks[1][2] == 3

    def test_multiple_files(self):
        diff = textwrap.dedent('''\
            diff --git a/a.py b/a.py
            --- a/a.py
            +++ b/a.py
            @@ -1,1 +1,2 @@
             x
            +y
            diff --git a/b.py b/b.py
            --- a/b.py
            +++ b/b.py
            @@ -5,1 +5,1 @@
            -old
            +new
            ''')
        hunks = _parse_unified_diff(diff)
        assert len(hunks) == 2
        assert hunks[0][0] == 'a.py'
        assert hunks[1][0] == 'b.py'
        assert hunks[1][1] == 5
