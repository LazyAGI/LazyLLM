# Tests for diff line-number annotation and large-hunk splitting logic.
import textwrap

from lazyllm.tools.git.review.utils import (
    _annotate_diff_with_line_numbers, _annotate_full_diff, _parse_unified_diff,
)
from lazyllm.tools.git.review.poster import _build_commentable_lines, _filter_commentable


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
        for win_start, _win_count, win_lines in wins:
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
# _split_file_diff_into_chunks: each chunk must carry a @@ header
# ---------------------------------------------------------------------------

class TestSplitFileChunks:
    def _make_big_hunk(self, new_start: int, n_lines: int) -> str:
        hdr = f'@@ -{new_start},{n_lines} +{new_start},{n_lines} @@\n'
        body = ''.join(f' line_{new_start + i:04d}\n' for i in range(n_lines))
        return hdr + body

    def test_small_diff_returns_single_chunk(self):
        from lazyllm.tools.git.review.rounds import _split_file_diff_into_chunks
        diff = self._make_big_hunk(100, 5)
        chunks = _split_file_diff_into_chunks(diff, 100000)
        assert len(chunks) == 1
        assert chunks[0][0] == 'all hunks'

    def test_each_chunk_has_hunk_header(self):
        from lazyllm.tools.git.review.rounds import _split_file_diff_into_chunks
        from lazyllm.tools.git.review.utils import _annotate_full_diff
        # Force splitting by using a tiny max_chars
        diff = self._make_big_hunk(100, 50)
        chunks = _split_file_diff_into_chunks(diff, 200)
        assert len(chunks) > 1, 'expected multiple chunks'
        for i, (_label, chunk) in enumerate(chunks):
            first_line = chunk.splitlines()[0] if chunk else ''
            assert first_line.startswith('@@'), (
                f'chunk[{i}] missing @@ header, starts with: {first_line!r}'
            )
            # annotate and verify first new-file line number is >= 100
            annotated = _annotate_full_diff(chunk)
            ann_lines = [ln for ln in annotated.splitlines() if ln.startswith('[')]
            if ann_lines:
                import re as _re
                m = _re.search(r'\|([\s\d]+)\]', ann_lines[0])
                new_no = int(m.group(1).strip()) if m else 0
                assert new_no >= 100, (
                    f'chunk[{i}] first new-file line={new_no}, expected >= 100'
                )

    def test_raises_when_split_needed_but_no_hunk_header(self):
        import pytest
        from lazyllm.tools.git.review.rounds import _split_file_diff_into_chunks
        # Raw hunk body without @@ header — splitting must raise.
        # Each line is 11 chars; max_chars=20 forces a split after the first line.
        body = ''.join(f' line_{i:04d}\n' for i in range(10))
        with pytest.raises(ValueError, match='missing hunk headers'):
            _split_file_diff_into_chunks(body, 20)

    def test_multi_hunk_chunks_have_correct_start(self):
        from lazyllm.tools.git.review.rounds import _split_file_diff_into_chunks
        from lazyllm.tools.git.review.utils import _annotate_full_diff
        # Two hunks: one at line 10, one at line 500
        diff = self._make_big_hunk(10, 5) + self._make_big_hunk(500, 5)
        chunks = _split_file_diff_into_chunks(diff, 100000)
        # Both hunks fit in one chunk
        assert len(chunks) == 1
        annotated = _annotate_full_diff(chunks[0][1])
        # line 10 and line 500 should both appear
        assert '|  10]' in annotated or '|  10]' in annotated
        assert '| 500]' in annotated or '| 500]' in annotated


# ---------------------------------------------------------------------------
# _annotate_full_diff: handles complete unified diff with @@ headers
# ---------------------------------------------------------------------------

class TestAnnotateFullDiff:
    def test_real_tools_md_case(self):
        # Reproduces the exact bug from PR #1082: tools.md hunk starts at line 545
        # but LLM was outputting line=6 (relative position in diff_chunk).
        diff = textwrap.dedent('''\
            @@ -545,3 +545,7 @@
             ::: lazyllm.tools.review.tools.chinese_corrector.ChineseCorrector
                 members: correct, correct_batch
                 exclude-members:
            +
            +::: lazyllm.tools.rag.QueryEnhACProcessor
            +    members:
            +    exclude-members:''')
        result = _annotate_full_diff(diff)
        lines = result.splitlines()
        # @@ header kept as-is
        assert lines[0].startswith('@@')
        # context lines start at 545
        assert '545' in lines[1] and '545' in lines[1]
        # first added line (blank) → new-file line 548
        assert '[--|' in lines[4] and '548' in lines[4]
        # +    members: → new-file line 550
        members_line = next(ln for ln in lines if 'members:' in ln and '[--|' in ln)
        assert '550' in members_line

    def test_hunk_header_resets_counters(self):
        diff = textwrap.dedent('''\
            @@ -1,2 +1,3 @@
             ctx1
            +add1
             ctx2
            @@ -10,2 +11,3 @@
             ctx3
            +add2
             ctx4''')
        result = _annotate_full_diff(diff)
        lines = result.splitlines()
        # first hunk: ctx1=1, add1→new=2, ctx2=2/3
        assert '|   1]' in lines[1] or '|  1]' in lines[1]
        # second hunk header resets; ctx3 should be at old=10, new=11
        hunk2_ctx = next(ln for ln in lines if 'ctx3' in ln)
        assert '10' in hunk2_ctx and '11' in hunk2_ctx
        # add2 → new=12
        add2_line = next(ln for ln in lines if 'add2' in ln)
        assert '[--|' in add2_line and '12' in add2_line

    def test_diff_git_headers_pass_through(self):
        diff = textwrap.dedent('''\
            diff --git a/foo.py b/foo.py
            index abc..def 100644
            --- a/foo.py
            +++ b/foo.py
            @@ -5,2 +5,3 @@
             line5
            +newline
             line6''')
        result = _annotate_full_diff(diff)
        lines = result.splitlines()
        assert lines[0] == 'diff --git a/foo.py b/foo.py'
        assert lines[1] == 'index abc..def 100644'
        assert lines[2] == '--- a/foo.py'
        assert lines[3] == '+++ b/foo.py'
        assert lines[4].startswith('@@')
        # +newline → new-file line 6
        new_line = next(ln for ln in lines if 'newline' in ln)
        assert '[--|' in new_line and '6' in new_line

    def test_deletion_no_new_line_number(self):
        diff = '@@ -3,3 +3,2 @@\n ctx\n-deleted\n ctx2'
        result = _annotate_full_diff(diff)
        del_line = next(ln for ln in result.splitlines() if 'deleted' in ln)
        assert '|--]' in del_line

    def test_empty_diff(self):
        assert _annotate_full_diff('') == ''

    def test_raises_on_added_line_without_hunk_header(self):
        import pytest
        with pytest.raises(ValueError, match='missing hunk headers'):
            _annotate_full_diff('+added line without @@ header')

    def test_raises_on_removed_line_without_hunk_header(self):
        import pytest
        with pytest.raises(ValueError, match='missing hunk headers'):
            _annotate_full_diff('-removed line without @@ header')

    def test_raises_on_context_line_without_hunk_header(self):
        import pytest
        with pytest.raises(ValueError, match='missing hunk headers'):
            _annotate_full_diff(' context line without @@ header')

    def test_file_metadata_before_hunk_header_is_ok(self):
        # diff/index/---/+++ lines before @@ are fine
        diff = 'diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,2 @@\n ctx\n+add'
        result = _annotate_full_diff(diff)
        assert '[--|   2]' in result


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


# ---------------------------------------------------------------------------
# _build_commentable_lines / _filter_commentable
# ---------------------------------------------------------------------------

class TestCommentableFilter:
    def _make_hunks(self, specs):
        # specs: list of (path, new_start, new_count)
        return [(p, s, c, '') for p, s, c in specs]

    def test_build_commentable_basic(self):
        hunks = self._make_hunks([('foo.py', 10, 5), ('bar.py', 1, 3)])
        cm = _build_commentable_lines(hunks)
        assert cm['foo.py'] == {10, 11, 12, 13, 14}
        assert cm['bar.py'] == {1, 2, 3}

    def test_build_commentable_multiple_hunks_same_file(self):
        hunks = self._make_hunks([('a.py', 1, 3), ('a.py', 20, 2)])
        cm = _build_commentable_lines(hunks)
        assert cm['a.py'] == {1, 2, 3, 20, 21}

    def test_filter_keeps_valid(self):
        hunks = self._make_hunks([('foo.py', 10, 5)])
        cm = _build_commentable_lines(hunks)
        comments = [
            {'path': 'foo.py', 'line': 10},
            {'path': 'foo.py', 'line': 14},
        ]
        kept, dropped = _filter_commentable(comments, cm)
        assert len(kept) == 2
        assert dropped == 0

    def test_filter_drops_out_of_range(self):
        hunks = self._make_hunks([('foo.py', 10, 5)])
        cm = _build_commentable_lines(hunks)
        comments = [
            {'path': 'foo.py', 'line': 9},   # before hunk
            {'path': 'foo.py', 'line': 15},  # after hunk (new_start+new_count = 15, exclusive)
            {'path': 'foo.py', 'line': 12},  # valid
        ]
        kept, dropped = _filter_commentable(comments, cm)
        assert len(kept) == 1
        assert kept[0]['line'] == 12
        assert dropped == 2

    def test_filter_drops_unknown_file(self):
        hunks = self._make_hunks([('foo.py', 1, 10)])
        cm = _build_commentable_lines(hunks)
        comments = [{'path': 'other.py', 'line': 5}]
        kept, dropped = _filter_commentable(comments, cm)
        assert kept == []
        assert dropped == 1

    def test_filter_drops_missing_line(self):
        hunks = self._make_hunks([('foo.py', 1, 10)])
        cm = _build_commentable_lines(hunks)
        comments = [{'path': 'foo.py'}]  # no 'line' key
        kept, dropped = _filter_commentable(comments, cm)
        assert kept == []
        assert dropped == 1

    def test_filter_empty_input(self):
        cm = _build_commentable_lines([])
        kept, dropped = _filter_commentable([], cm)
        assert kept == [] and dropped == 0

    # --- precise content-based parsing (non-empty content) ---

    def test_build_commentable_with_content_excludes_deleted_lines(self):
        # hunk: -del1, +add1, ' ctx' → new-file lines: add1=10, ctx=11; del1 has no new-file line
        content = '-del1\n+add1\n ctx'
        hunks = [('f.py', 10, 0, content)]
        cm = _build_commentable_lines(hunks)
        assert cm['f.py'] == {10, 11}
        assert 9 not in cm['f.py']   # before hunk
        assert 12 not in cm['f.py']  # beyond content

    def test_build_commentable_with_content_pure_deletion(self):
        # pure-delete hunk: all '-' lines → no new-file lines at all
        content = '-gone1\n-gone2\n-gone3'
        hunks = [('f.py', 5, 0, content)]
        cm = _build_commentable_lines(hunks)
        assert cm['f.py'] == set()

    def test_build_commentable_with_content_mixed_hunk(self):
        # Simulates @@ -16,6 +16,8 @@
        # ' R1', ' R2', ' R3', '+R4A', ' R4', ' FINAL', '+UPLOAD'
        content = " R1 = 'r1'\n R2 = 'r2'\n R3 = 'r3'\n+R4A = 'r4a'\n R4 = 'r4'\n FINAL = 'final'\n+UPLOAD = 'upload'"
        hunks = [('f.py', 16, 0, content)]
        cm = _build_commentable_lines(hunks)
        # context lines: 16,17,18,20,21; added lines: 19,22
        assert cm['f.py'] == {16, 17, 18, 19, 20, 21, 22}

    def test_filter_drops_deleted_line_number_with_content(self):
        # '-' line at new_start should NOT be commentable
        content = '-deleted\n+added\n ctx'
        hunks = [('f.py', 10, 0, content)]
        cm = _build_commentable_lines(hunks)
        # 'deleted' is a '-' line → no new-file number; 'added'→10, 'ctx'→11
        comments = [
            {'path': 'f.py', 'line': 10},   # valid (added line)
            {'path': 'f.py', 'line': 11},   # valid (context line)
        ]
        kept, dropped = _filter_commentable(comments, cm)
        assert len(kept) == 2
        assert dropped == 0

    def test_deleted_and_added_at_same_position(self):
        # '-' and '+' at the same visual position: the new-file line number belongs
        # to the '+' line, not the '-' line.  Both share "line 10" visually, but
        # only the '+' line exists in the new file.
        # diff: ' ctx'=10, '-old'=no new-no, '+new'=11, ' ctx2'=12
        content = ' ctx\n-old_line\n+new_line\n ctx2'
        hunks = [('f.py', 10, 0, content)]
        cm = _build_commentable_lines(hunks)
        # ctx→10, new_line→11, ctx2→12; old_line has no new-file number
        assert cm['f.py'] == {10, 11, 12}
        # line 11 is the '+new_line', not '-old_line'
        kept, dropped = _filter_commentable([{'path': 'f.py', 'line': 11}], cm)
        assert len(kept) == 1 and dropped == 0

    def test_filter_drops_line_beyond_content_with_content(self):
        content = '+add1\n ctx'   # new-file lines 5, 6
        hunks = [('f.py', 5, 0, content)]
        cm = _build_commentable_lines(hunks)
        comments = [
            {'path': 'f.py', 'line': 7},   # beyond content
            {'path': 'f.py', 'line': 4},   # before hunk
            {'path': 'f.py', 'line': 5},   # valid
        ]
        kept, dropped = _filter_commentable(comments, cm)
        assert len(kept) == 1
        assert kept[0]['line'] == 5
        assert dropped == 2
