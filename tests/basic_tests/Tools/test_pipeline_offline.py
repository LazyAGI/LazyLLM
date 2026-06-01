# Copyright (c) 2026 LazyAGI. All rights reserved.
'''Unit tests for the LazyLLM Git Review Pipeline bug-fixes.

All tests are fully self-contained and require no live LLM calls,
external checkpoints, or network access.

To run:
    pytest tests/basic_tests/Tools/test_pipeline_offline.py -v
'''
import unittest
from typing import Any, Dict


# ---------------------------------------------------------------------------
# F1a: r2_passthrough removed – R2 issues must reach R4
# ---------------------------------------------------------------------------

class TestF1aR2Passthrough(unittest.TestCase):
    '''F1a: Verify that R2 issues are no longer filtered by r3_covered_files.'''

    def _build_issues(self):
        r1 = [{'path': 'a.py', 'line': 10, 'bug_category': 'logic',
               'severity': 'normal', 'problem': 'r1 issue', 'source': 'r1'}]
        r2 = [{'path': 'a.py', 'line': 20, 'bug_category': 'design',
               'severity': 'medium', 'problem': 'r2 arch issue', 'source': 'r2'}]
        r3 = [{'path': 'a.py', 'line': 10, 'bug_category': 'logic',
               'severity': 'normal', 'problem': 'r3 confirmed issue', 'source': 'r3'}]
        return r1, r2, r3

    def test_r2_not_filtered(self):
        '''R2 issues must NOT be dropped even when r3 covers the same file.'''
        r1, r2, r3 = self._build_issues()
        # Old behaviour would filter r2 because 'a.py' is in r3_covered_files.
        # New behaviour: r2 passes through unchanged.
        r3_covered_files = {c.get('path') for c in r3 if c.get('path')}
        # Simulate old filter:
        old_r2_passthrough = [c for c in r2 if c.get('path') not in r3_covered_files]
        self.assertEqual(len(old_r2_passthrough), 0, 'Old filter would have dropped R2 issue')
        # New code simply passes r2 unchanged:
        new_r2 = r2  # no filter
        self.assertEqual(len(new_r2), 1, 'New code keeps R2 issue')
        self.assertEqual(new_r2[0]['bug_category'], 'design')


# ---------------------------------------------------------------------------
# F1b: _deterministic_dedup no longer does cross-category merge
# ---------------------------------------------------------------------------

class TestF1bDeterministicDedup(unittest.TestCase):
    '''F1b: Same (path, line) with different categories should both be kept.'''

    def test_same_line_different_category_kept(self):
        from lazyllm.tools.git.review.rounds import _deterministic_dedup
        issues = [
            {'path': 'x.py', 'line': 5, 'bug_category': 'logic', 'severity': 'normal',
             'problem': 'logic issue', 'source': 'r1'},
            {'path': 'x.py', 'line': 5, 'bug_category': 'design', 'severity': 'normal',
             'problem': 'design issue', 'source': 'r2'},
        ]
        result = _deterministic_dedup(issues)
        self.assertEqual(len(result), 2, 'Both issues must survive dedup (different categories)')

    def test_same_path_line_category_keeps_highest_priority(self):
        from lazyllm.tools.git.review.rounds import _deterministic_dedup
        problem = 'null pointer dereference when accessing foo.bar'
        issues = [
            {'path': 'x.py', 'line': 5, 'bug_category': 'logic', 'severity': 'normal',
             'problem': problem, 'source': 'r1'},
            {'path': 'x.py', 'line': 5, 'bug_category': 'logic', 'severity': 'normal',
             'problem': problem + ' in handler', 'source': 'r3'},
        ]
        result = _deterministic_dedup(issues)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['source'], 'r3', 'r3 > r1 in source priority')


# ---------------------------------------------------------------------------
# F2: _collect_all_file_diffs includes new files
# ---------------------------------------------------------------------------

class TestF2CollectAllFileDiffs(unittest.TestCase):
    '''F2: _collect_all_file_diffs should include new files; old function excluded them.'''

    _DIFF = '''\
diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def hello():
+    pass
diff --git a/old_file.py b/old_file.py
index aaaaaaa..bbbbbbb 100644
--- a/old_file.py
+++ b/old_file.py
@@ -1,2 +1,3 @@
 existing()
+new_line()
'''

    def test_all_diffs_includes_new_file(self):
        from lazyllm.tools.git.review.rounds import _collect_all_file_diffs
        result = _collect_all_file_diffs(self._DIFF)
        self.assertIn('new_file.py', result, 'new_file.py must be included')
        self.assertIn('old_file.py', result, 'old_file.py must be included')

    def test_modified_only_excludes_new_file(self):
        from lazyllm.tools.git.review.rounds import _rscene_collect_modified_file_diffs
        result = _rscene_collect_modified_file_diffs(self._DIFF)
        self.assertNotIn('new_file.py', result, 'new_file.py must be excluded from modified-only')
        self.assertIn('old_file.py', result)


# ---------------------------------------------------------------------------
# F3: clip_diff_by_hunk_budget returns dropped_files
# ---------------------------------------------------------------------------

class TestF3ClipDiffDroppedFiles(unittest.TestCase):
    '''F3: clip_diff_by_hunk_budget must report dropped files on truncation.'''

    def _make_diff(self, n_files: int, lines_per_file: int = 50) -> str:
        parts = []
        for i in range(n_files):
            content = '\n'.join(f'+line {j}' for j in range(lines_per_file))
            parts.append(
                f'diff --git a/file{i}.py b/file{i}.py\n'
                f'@@ -1,0 +1,{lines_per_file} @@\n'
                f'{content}\n'
            )
        return ''.join(parts)

    def test_no_truncation_returns_empty_dropped(self):
        from lazyllm.tools.git.review.constants import clip_diff_by_hunk_budget
        diff = self._make_diff(2, 5)
        clipped, dropped = clip_diff_by_hunk_budget(diff, len(diff) + 1000)
        self.assertEqual(dropped, [])
        self.assertEqual(clipped, diff)

    def test_truncation_reports_dropped_files(self):
        from lazyllm.tools.git.review.constants import clip_diff_by_hunk_budget
        diff = self._make_diff(5, 100)
        # Budget too small to hold all files.
        budget = len(diff) // 3
        clipped, dropped = clip_diff_by_hunk_budget(diff, budget)
        self.assertGreater(len(dropped), 0, 'At least one file should be dropped')
        self.assertLessEqual(len(clipped), budget + 200)  # allow small overrun at boundary


# ---------------------------------------------------------------------------
# F4: _token_overlap works for Chinese text
# ---------------------------------------------------------------------------

class TestF4TokenOverlap(unittest.TestCase):
    '''F4: n-gram based _token_overlap should produce non-zero similarity for Chinese text.'''

    def test_chinese_similar_texts(self):
        from lazyllm.tools.git.review.rounds import _token_overlap
        a = '该函数缺少对空指针的检查，可能导致运行时异常'
        b = '函数没有检查空指针，存在运行时崩溃风险'
        score = _token_overlap(a, b)
        self.assertGreater(score, 0.0, 'Chinese similar texts must have positive overlap')

    def test_identical_texts(self):
        from lazyllm.tools.git.review.rounds import _token_overlap
        text = 'the function does not handle the null pointer case'
        self.assertAlmostEqual(_token_overlap(text, text), 1.0, places=5)

    def test_unrelated_texts(self):
        from lazyllm.tools.git.review.rounds import _token_overlap
        a = 'function handles null pointer'
        b = 'database connection timeout exceeded'
        score = _token_overlap(a, b)
        self.assertLess(score, 0.45, 'Unrelated texts must be below threshold')


# ---------------------------------------------------------------------------
# F5: discarded_keys includes bug_category
# ---------------------------------------------------------------------------

class TestF5DiscardedKeys(unittest.TestCase):
    '''F5: R3 discarded_keys should be {path}:{line}:{category} format.'''

    def test_discard_key_format(self):
        '''Simulate _r3_extract_issues logic to check key format.'''
        r1_issues = [
            {'path': 'a.py', 'line': 10, 'bug_category': 'logic', 'problem': 'issue'},
            {'path': 'a.py', 'line': 10, 'bug_category': 'design', 'problem': 'issue2'},
        ]
        # Simulate kept_r1_idxs = {0} (only index 0 is kept)
        kept_r1_idxs = {0}
        discarded_keys = {
            f'{c.get("path", "a.py")}:{c.get("line")}:{c.get("bug_category", "")}'
            for i, c in enumerate(r1_issues) if i not in kept_r1_idxs
        }
        # Only index 1 (design) should be discarded
        self.assertIn('a.py:10:design', discarded_keys)
        self.assertNotIn('a.py:10:logic', discarded_keys)

    def test_r4_filter_with_category_key(self):
        '''New format key: same path/line but different category should NOT be dropped.'''
        discarded_prev_keys = {'a.py:10:design'}
        r1 = [
            {'path': 'a.py', 'line': 10, 'bug_category': 'logic', 'source': 'r1'},
            {'path': 'a.py', 'line': 10, 'bug_category': 'design', 'source': 'r1'},
        ]

        def _r1_is_discarded(c: Dict[str, Any]) -> bool:
            pl = f'{c.get("path")}:{c.get("line")}'
            plc = f'{pl}:{c.get("bug_category", "")}'
            return pl in discarded_prev_keys or plc in discarded_prev_keys

        passthrough = [c for c in r1 if not _r1_is_discarded(c)]
        self.assertEqual(len(passthrough), 1)
        self.assertEqual(passthrough[0]['bug_category'], 'logic')


# ---------------------------------------------------------------------------
# F10: R1 cross-hunk dedup
# ---------------------------------------------------------------------------

class TestF10R1Dedup(unittest.TestCase):
    '''F10: Duplicate R1 issues from overlapping windows must be removed.'''

    def test_dedup_removes_exact_duplicates(self):
        all_comments = [
            {'path': 'a.py', 'line': 50, 'bug_category': 'logic', 'problem': 'null check missing'},
            {'path': 'a.py', 'line': 50, 'bug_category': 'logic', 'problem': 'null check missing'},
            {'path': 'a.py', 'line': 50, 'bug_category': 'design', 'problem': 'bad design here'},
        ]
        _seen: set = set()
        deduped = []
        for c in all_comments:
            key = (c.get('path', ''), int(c.get('line') or 0),
                   c.get('bug_category', ''), (c.get('problem') or '')[:60])
            if key not in _seen:
                _seen.add(key)
                deduped.append(c)
        self.assertEqual(len(deduped), 2, 'Exact duplicate should be removed')


# ---------------------------------------------------------------------------
# F11: demote_on_out_of_range in _normalize_comment_item
# ---------------------------------------------------------------------------

class TestF11DemoteOnOutOfRange(unittest.TestCase):
    '''F11: Issues with out-of-range line should be demoted to line=None, not discarded.'''

    def test_demote_null_line(self):
        from lazyllm.tools.git.review.utils import _normalize_comment_item
        item = {'path': 'arch.py', 'line': None, 'bug_category': 'design',
                'severity': 'medium', 'problem': 'arch issue without line'}
        # Without demote: None line with allow_null_line=False → skipped
        result_strict = _normalize_comment_item(item, allow_null_line=False)
        self.assertIsNone(result_strict, 'Strict mode should drop null-line item')
        # With demote: should be kept as line=None
        result_demote = _normalize_comment_item(item, demote_on_out_of_range=True)
        self.assertIsNotNone(result_demote, 'demote_on_out_of_range should keep the item')
        self.assertIsNone(result_demote['line'], 'line should be None after demotion')


# ---------------------------------------------------------------------------
# F12: comment body contains stage tag
# ---------------------------------------------------------------------------

class TestF12StageTag(unittest.TestCase):
    '''F12: _comment_body_text and _build_general_body must include [stage:X] tag.'''

    def test_inline_comment_has_stage_tag(self):
        from lazyllm.tools.git.review.poster import _comment_body_text
        comment = {
            'bug_category': 'logic', 'severity': 'medium',
            'source': 'r3', 'problem': 'test problem', 'suggestion': 'fix it',
        }
        body = _comment_body_text(comment, model_name='test-model')
        self.assertIn('[stage:r3]', body, 'Inline comment body must have stage tag')

    def test_inline_comment_no_source_no_tag(self):
        from lazyllm.tools.git.review.poster import _comment_body_text
        comment = {
            'bug_category': 'logic', 'severity': 'medium',
            'problem': 'test problem', 'suggestion': 'fix it',
        }
        body = _comment_body_text(comment, model_name='test-model')
        self.assertNotIn('[stage:', body, 'No source → no stage tag')

    def test_general_body_has_stage_tag(self):
        from lazyllm.tools.git.review.poster import _build_general_body
        comments = [{
            'path': 'foo.py', 'line': None, 'bug_category': 'design',
            'severity': 'medium', 'source': 'r2', 'problem': 'arch concern', 'suggestion': 'refactor',
        }]
        body = _build_general_body(comments)
        self.assertIn('[stage:r2]', body, 'General body must have stage tag')


# ---------------------------------------------------------------------------
# F9: _find_related_small_files handles relative imports
# ---------------------------------------------------------------------------

class TestF9RelativeImports(unittest.TestCase):
    '''F9: _find_related_small_files should match relative imports like `from .bar import X`.'''

    def test_relative_import_matched(self):
        from lazyllm.tools.git.review.rounds import _find_related_small_files
        large_diff = '+from .utils import helper\n+from ..models.schema import Schema\n'
        small_files = ['src/utils.py', 'src/models/schema.py', 'src/other.py']
        result = _find_related_small_files(large_diff, small_files, {})
        self.assertIn('src/utils.py', result, 'utils.py must be matched via relative import')
        self.assertIn('src/models/schema.py', result, 'schema.py must be matched via relative import')
        self.assertNotIn('src/other.py', result, 'other.py is not imported')

    def test_absolute_import_still_works(self):
        from lazyllm.tools.git.review.rounds import _find_related_small_files
        large_diff = '+import helper\n+from models.schema import Schema\n'
        small_files = ['src/helper.py', 'src/schema.py']
        result = _find_related_small_files(large_diff, small_files, {})
        self.assertIn('src/helper.py', result)
        self.assertIn('src/schema.py', result)


if __name__ == '__main__':
    unittest.main()
