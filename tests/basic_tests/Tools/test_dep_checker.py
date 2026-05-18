# Copyright (c) 2026 LazyAGI. All rights reserved.
'''Tests for dep_checker: import extraction, cycle detection, inversion detection.'''
import os
import tempfile
import textwrap
import unittest


class TestPythonExtractor(unittest.TestCase):
    '''Test Python import extraction (AST + regex fallback).'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # create a mini project structure
        os.makedirs(os.path.join(self.tmpdir, 'pkg', 'sub'), exist_ok=True)
        self._write('pkg/__init__.py', '')
        self._write('pkg/sub/__init__.py', '')
        self._write('pkg/alpha.py', 'class Alpha: pass')
        self._write('pkg/beta.py', 'class Beta: pass')
        self._write('pkg/sub/gamma.py', 'class Gamma: pass')

    def _write(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)

    def test_absolute_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_python
        content = 'from pkg.alpha import Alpha\nimport pkg.beta'
        edges = extract_imports_python('main.py', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('pkg/alpha.py', targets)
        self.assertIn('pkg/beta.py', targets)

    def test_relative_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_python
        # from .alpha means pkg/sub/alpha.py (same package)
        self._write('pkg/sub/alpha.py', 'class SubAlpha: pass')
        content = 'from .alpha import SubAlpha\nfrom .. import alpha'
        # file is pkg/sub/gamma.py, so . = pkg/sub, .. = pkg
        edges = extract_imports_python('pkg/sub/gamma.py', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('pkg/sub/alpha.py', targets)
        self.assertIn('pkg/alpha.py', targets)

    def test_syntax_error_fallback(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_python
        content = 'from pkg.alpha import Alpha\ndef broken(:\n  pass'
        edges = extract_imports_python('main.py', content, self.tmpdir)
        self.assertTrue(len(edges) >= 1)
        self.assertEqual(edges[0].target, 'pkg/alpha.py')

    def test_third_party_skipped(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_python
        content = 'import numpy\nfrom requests import get'
        edges = extract_imports_python('main.py', content, self.tmpdir)
        self.assertEqual(len(edges), 0)

    def test_init_package_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_python
        content = 'from pkg.sub import gamma'
        edges = extract_imports_python('main.py', content, self.tmpdir)
        targets = {e.target for e in edges}
        # should resolve to pkg/sub/__init__.py or pkg/sub/gamma.py
        self.assertTrue(
            'pkg/sub/__init__.py' in targets or 'pkg/sub/gamma.py' in targets,
            f'Expected pkg/sub resolution, got {targets}',
        )


class TestJsExtractor(unittest.TestCase):
    '''Test JS/TS import extraction.'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'src', 'components'), exist_ok=True)
        self._write('src/utils.ts', 'export const foo = 1;')
        self._write('src/components/Button.tsx', 'export default function Button() {}')
        self._write('src/components/index.ts', 'export { default as Button } from "./Button";')

    def _write(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)

    def test_relative_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_js
        content = "import { foo } from '../utils';\nimport Button from './Button';"
        edges = extract_imports_js('src/components/App.tsx', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('src/utils.ts', targets)
        self.assertIn('src/components/Button.tsx', targets)

    def test_require(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_js
        content = "const utils = require('../utils');"
        edges = extract_imports_js('src/components/App.tsx', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('src/utils.ts', targets)

    def test_node_modules_skipped(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_js
        content = "import React from 'react';\nimport { useState } from 'react';"
        edges = extract_imports_js('src/App.tsx', content, self.tmpdir)
        self.assertEqual(len(edges), 0)

    def test_directory_index(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_js
        content = "import { Button } from './components';"
        edges = extract_imports_js('src/App.tsx', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('src/components/index.ts', targets)


class TestGoExtractor(unittest.TestCase):
    '''Test Go import extraction.'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'internal', 'handler'), exist_ok=True)
        self._write('go.mod', 'module github.com/example/myapp\n\ngo 1.21\n')
        self._write('internal/handler/handler.go', 'package handler')

    def _write(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)

    def test_single_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_go
        content = 'package main\n\nimport "github.com/example/myapp/internal/handler"\n'
        edges = extract_imports_go('cmd/main.go', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('internal/handler', targets)

    def test_block_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_go
        content = textwrap.dedent('''\
            package main

            import (
                "fmt"
                "github.com/example/myapp/internal/handler"
            )
        ''')
        edges = extract_imports_go('cmd/main.go', content, self.tmpdir)
        targets = {e.target for e in edges}
        self.assertIn('internal/handler', targets)
        # stdlib "fmt" should be skipped
        self.assertNotIn('fmt', targets)

    def test_external_dep_skipped(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_go
        content = 'package main\n\nimport "github.com/other/lib"\n'
        edges = extract_imports_go('cmd/main.go', content, self.tmpdir)
        self.assertEqual(len(edges), 0)


class TestJavaExtractor(unittest.TestCase):
    '''Test Java import extraction.'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'src', 'main', 'java', 'com', 'example'), exist_ok=True)
        self._write('src/main/java/com/example/Service.java', 'package com.example;')

    def _write(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)

    def test_java_import(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_java
        content = 'package com.example;\n\nimport com.example.Service;\n'
        edges = extract_imports_java(
            'src/main/java/com/example/Controller.java', content, self.tmpdir,
        )
        targets = {e.target for e in edges}
        self.assertIn('src/main/java/com/example/Service.java', targets)

    def test_external_skipped(self):
        from lazyllm.tools.git.review.dep_checker import extract_imports_java
        content = 'import org.springframework.web.bind.annotation.RestController;\n'
        edges = extract_imports_java(
            'src/main/java/com/example/Controller.java', content, self.tmpdir,
        )
        self.assertEqual(len(edges), 0)


class TestCycleDetection(unittest.TestCase):
    '''Test cycle detection at file and module level.'''

    def test_simple_cycle(self):
        from lazyllm.tools.git.review.dep_checker import _find_cycles_dfs
        graph = {
            'a.py': {'b.py'},
            'b.py': {'a.py'},
        }
        new_edges = {('b.py', 'a.py')}
        cycles = _find_cycles_dfs(graph, new_edges)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(cycles[0]), 3)  # [a, b, a] or [b, a, b]

    def test_three_node_cycle(self):
        from lazyllm.tools.git.review.dep_checker import _find_cycles_dfs
        graph = {
            'a.py': {'b.py'},
            'b.py': {'c.py'},
            'c.py': {'a.py'},
        }
        new_edges = {('c.py', 'a.py')}
        cycles = _find_cycles_dfs(graph, new_edges)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(cycles[0]), 4)  # [a, b, c, a]

    def test_old_cycle_not_reported(self):
        from lazyllm.tools.git.review.dep_checker import _find_cycles_dfs
        graph = {
            'a.py': {'b.py'},
            'b.py': {'a.py'},
        }
        new_edges = set()  # no new edges
        cycles = _find_cycles_dfs(graph, new_edges)
        self.assertEqual(len(cycles), 0)

    def test_no_cycle(self):
        from lazyllm.tools.git.review.dep_checker import _find_cycles_dfs
        graph = {
            'a.py': {'b.py'},
            'b.py': {'c.py'},
            'c.py': set(),
        }
        new_edges = {('a.py', 'b.py')}
        cycles = _find_cycles_dfs(graph, new_edges)
        self.assertEqual(len(cycles), 0)

    def test_long_cycle_capped(self):
        from lazyllm.tools.git.review.dep_checker import _find_cycles_dfs
        # 6-node cycle should be skipped (max_length=5)
        nodes = [f'{i}.py' for i in range(6)]
        graph = {nodes[i]: {nodes[(i + 1) % 6]} for i in range(6)}
        new_edges = {(nodes[5], nodes[0])}
        cycles = _find_cycles_dfs(graph, new_edges, max_length=5)
        self.assertEqual(len(cycles), 0)


class TestModuleAggregation(unittest.TestCase):
    '''Test module-level graph aggregation.'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for pkg in ['pkg/rag', 'pkg/agent']:
            d = os.path.join(self.tmpdir, pkg)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, '__init__.py'), 'w') as f:
                f.write('')

    def test_aggregation(self):
        from lazyllm.tools.git.review.dep_checker import aggregate_to_module_graph
        file_graph = {
            'pkg/rag/base.py': {'pkg/agent/react.py'},
            'pkg/agent/react.py': {'pkg/rag/base.py'},
        }
        mod_graph, f2m = aggregate_to_module_graph(file_graph, self.tmpdir)
        self.assertIn('pkg/rag', mod_graph)
        self.assertIn('pkg/agent', mod_graph.get('pkg/rag', set()))
        self.assertIn('pkg/rag', mod_graph.get('pkg/agent', set()))

    def test_same_module_edges_ignored(self):
        from lazyllm.tools.git.review.dep_checker import aggregate_to_module_graph
        file_graph = {
            'pkg/rag/base.py': {'pkg/rag/utils.py'},
            'pkg/rag/utils.py': set(),
        }
        mod_graph, _ = aggregate_to_module_graph(file_graph, self.tmpdir)
        self.assertEqual(mod_graph.get('pkg/rag', set()), set())


class TestInversionDetection(unittest.TestCase):
    '''Test dependency inversion detection.'''

    def test_heuristic_inversion(self):
        from lazyllm.tools.git.review.dep_checker import detect_inversions
        tmpdir = tempfile.mkdtemp()
        graph = {
            'core/base.py': {'api/handler.py'},
            'api/handler.py': set(),
        }
        new_edges = {('core/base.py', 'api/handler.py')}
        issues = detect_inversions(graph, new_edges, tmpdir)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['source'], 'core/base.py')
        self.assertEqual(issues[0]['target'], 'api/handler.py')

    def test_correct_direction_no_issue(self):
        from lazyllm.tools.git.review.dep_checker import detect_inversions
        tmpdir = tempfile.mkdtemp()
        graph = {
            'api/handler.py': {'core/base.py'},
            'core/base.py': set(),
        }
        new_edges = {('api/handler.py', 'core/base.py')}
        issues = detect_inversions(graph, new_edges, tmpdir)
        self.assertEqual(len(issues), 0)

    def test_unknown_layer_skipped(self):
        from lazyllm.tools.git.review.dep_checker import detect_inversions
        tmpdir = tempfile.mkdtemp()
        graph = {
            'foo/bar.py': {'baz/qux.py'},
            'baz/qux.py': set(),
        }
        new_edges = {('foo/bar.py', 'baz/qux.py')}
        issues = detect_inversions(graph, new_edges, tmpdir)
        self.assertEqual(len(issues), 0)


class TestRunDepAnalysis(unittest.TestCase):
    '''Integration test for _run_dep_analysis.'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'pkg'), exist_ok=True)
        self._write('pkg/__init__.py', '')
        self._write('pkg/a.py', 'from pkg.b import B\nclass A: pass')
        self._write('pkg/b.py', 'from pkg.a import A\nclass B: pass')

    def _write(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)

    def test_detects_cycle_from_diff(self):
        from lazyllm.tools.git.review.dep_checker import _run_dep_analysis
        diff = textwrap.dedent('''\
            diff --git a/pkg/b.py b/pkg/b.py
            --- a/pkg/b.py
            +++ b/pkg/b.py
            @@ -1,1 +1,2 @@
            +from pkg.a import A
             class B: pass
        ''')
        issues = _run_dep_analysis(diff, self.tmpdir)
        self.assertTrue(len(issues) >= 1)
        cycle_issues = [i for i in issues if 'ircular' in i.get('problem', '')]
        self.assertTrue(len(cycle_issues) >= 1)

    def test_no_issues_on_clean_diff(self):
        from lazyllm.tools.git.review.dep_checker import _run_dep_analysis
        self._write('pkg/c.py', 'class C: pass')
        diff = textwrap.dedent('''\
            diff --git a/pkg/c.py b/pkg/c.py
            --- /dev/null
            +++ b/pkg/c.py
            @@ -0,0 +1,1 @@
            +class C: pass
        ''')
        issues = _run_dep_analysis(diff, self.tmpdir)
        self.assertEqual(len(issues), 0)


class TestMultiLineMerge(unittest.TestCase):
    '''Test that multiple imports to the same target are merged into one issue.'''

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'core'), exist_ok=True)
        os.makedirs(os.path.join(self.tmpdir, 'api'), exist_ok=True)
        self._write('core/__init__.py', '')
        self._write('api/__init__.py', '')
        self._write('api/handler.py', 'class Handler: pass')
        # core/base.py imports api/handler.py on multiple lines
        self._write('core/base.py', (
            'from api.handler import Handler\n'
            'from api.handler import Handler as H2\n'
            'from api.handler import Handler as H3\n'
            'class Base: pass\n'
        ))

    def _write(self, rel_path, content):
        full = os.path.join(self.tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)

    def test_inversion_merged_lines(self):
        from lazyllm.tools.git.review.dep_checker import _run_dep_analysis
        diff = textwrap.dedent('''\
            diff --git a/core/base.py b/core/base.py
            --- /dev/null
            +++ b/core/base.py
            @@ -0,0 +1,4 @@
            +from api.handler import Handler
            +from api.handler import Handler as H2
            +from api.handler import Handler as H3
            +class Base: pass
        ''')
        issues = _run_dep_analysis(diff, self.tmpdir)
        inv_issues = [i for i in issues if 'inversion' in i.get('problem', '').lower()]
        # should be exactly 1 merged issue, not 3
        self.assertEqual(len(inv_issues), 1)
        problem = inv_issues[0]['problem']
        # should mention the additional lines
        self.assertIn('also at line', problem)
        # primary line should be 1
        self.assertEqual(inv_issues[0]['line'], 1)


if __name__ == '__main__':
    unittest.main()
