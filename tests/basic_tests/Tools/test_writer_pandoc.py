import shutil
import subprocess
from pathlib import Path

import pytest

from lazyllm.tools.writer.utils import conversion
from lazyllm.tools.writer.utils.pandoc import (
    PandocError,
    build_pandoc_command,
    check_pandoc_version,
    normalize_writer_markdown_for_latex,
    run_pandoc,
)


def _completed(command, *, stdout='', stderr='', returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def _pandoc_paths():
    pandoc_path = shutil.which('pandoc')
    if pandoc_path is None:
        pytest.skip('Pandoc is not installed.')
    version = subprocess.run(
        [pandoc_path, '--version'], capture_output=True, text=True, check=False,
    ).stdout.splitlines()[0]
    if version != 'pandoc 3.11':
        pytest.skip(f'Pandoc 3.11 is required, found {version!r}.')
    template_dir = (
        Path(__file__).resolve().parents[3]
        / 'lazyllm/tools/writer/templates/latex'
    )
    return pandoc_path, template_dir, template_dir / 'filters/writer.lua'


def _run_filter(markdown: str, *, template_name: str | None = None):
    pandoc_path, template_dir, filter_path = _pandoc_paths()
    command = [
        pandoc_path, '--from=gfm-yaml_metadata_block', '--to=latex',
        '--number-sections', '--lua-filter', str(filter_path),
    ]
    if template_name is not None:
        command.extend(('--standalone', '--template', str(template_dir / template_name)))
    return subprocess.run(
        command, input=markdown, capture_output=True, text=True, check=False,
    )


def test_pandoc_execution_contract(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if '--version' in command:
            return _completed(command, stdout='pandoc 3.11\n')
        return _completed(command, stdout='\\section{Title}\n')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert check_pandoc_version('/runtime/pandoc', required_version='3.11') == '3.11'
    command = build_pandoc_command(
        '/runtime/pandoc', '/runtime/basic_zh.tex', '/runtime/writer.lua',
    )
    assert command == [
        '/runtime/pandoc', '--from=gfm-yaml_metadata_block', '--to=latex',
        '--standalone', '--number-sections', '--wrap=none', '--template',
        '/runtime/basic_zh.tex', '--lua-filter', '/runtime/writer.lua',
    ]
    assert run_pandoc('# Title', command) == '\\section{Title}\n'
    assert calls[-1][1]['input'] == '# Title'
    assert calls[-1][1]['shell'] is False


def test_pandoc_execution_enforces_limits(monkeypatch):
    with pytest.raises(PandocError, match='input exceeds') as raised:
        run_pandoc('中文', ['pandoc'], max_input_bytes=5)
    assert raised.value.code == 'PANDOC_INPUT_TOO_LARGE'

    monkeypatch.setattr(
        subprocess, 'run',
        lambda command, **kwargs: _completed(command, stdout='123456'),
    )
    with pytest.raises(PandocError, match='output exceeds') as raised:
        run_pandoc('ok', ['pandoc'], max_output_bytes=5)
    assert raised.value.code == 'PANDOC_OUTPUT_TOO_LARGE'


def test_normalize_writer_markdown_builds_structured_anchors():
    markdown = (
        '# Document\n\n'
        '<a id="block-intro"></a>\n## 1. Introduction\n\n'
        '<a id="block-figure"></a>\n'
        '![Figure 1 Architecture](/tmp/images/arch.png)\n\n'
        '表1：实验结果 & 对比\n| A | B |\n|---|---|\n| 1 | 2 |\n\n'
        'Code 1: Query flow\n```python\nprint("ok")\n```\n\n'
        '<a id="block-formula"></a>\n$$\nx+y=z\n$$\n'
    )

    normalized = normalize_writer_markdown_for_latex(markdown)

    assert 'id="block-intro" data-kind="section" data-caption="Introduction"' in normalized
    assert 'id="block-figure" data-kind="figure" data-caption="Architecture"' in normalized
    assert (
        'id="block-table-001" data-kind="table" data-caption="实验结果 &amp; 对比"'
        in normalized
    )
    assert 'id="block-code-001" data-kind="code" data-caption="Query flow"' in normalized
    assert 'id="block-formula" data-kind="equation"' in normalized
    assert '## Introduction' in normalized
    assert '\n表1' not in normalized
    assert '\nCode 1:' not in normalized
    assert normalize_writer_markdown_for_latex(normalized) == normalized


def test_normalize_writer_markdown_preserves_canonical_titles_and_unnumbered_math():
    normalized = normalize_writer_markdown_for_latex(
        '# Document\n\n## 1. User-authored title\n\n$$\nx+y=z\n$$\n',
        materialized_numbering=False,
    )

    assert 'data-caption="1. User-authored title"' in normalized
    assert '## 1. User-authored title' in normalized
    assert 'data-kind="equation"' not in normalized


def test_normalize_numbered_bibliography_by_structure():
    normalized = normalize_writer_markdown_for_latex(
        '# Document\n\n正文引用\\[2]和\\[1]。\n\n'
        '## Sources consulted\n\n'
        '\\[1] First reference [J].\n\n'
        '\\[2] Second reference [M].\n'
    )

    assert '[\\[1\\]](#writer-cite-ref-001)' in normalized
    assert '[\\[2\\]](#writer-cite-ref-002)' in normalized
    assert 'data-writer-bibliography="begin" data-title="Sources consulted"' in normalized
    assert 'data-writer-bibitem="ref-001"' in normalized
    assert 'data-writer-bibitem="ref-002"' in normalized


def test_writer_conversion_routes_markdown_to_latex(monkeypatch):
    calls = []

    def convert(content, *, language='zh-CN', materialized_numbering=True):
        calls.append((content, language, materialized_numbering))
        return 'latex'

    monkeypatch.setattr(conversion, 'markdown_to_latex', convert)

    assert conversion.convert_writer_content(
        '# Title', 'markdown', 'latex', language='en-US',
        materialized_numbering=False,
    ) == 'latex'
    assert calls == [('# Title', 'en-US', False)]
    with pytest.raises(ValueError, match='only supports Markdown'):
        conversion.convert_writer_content('{}', 'lmd', 'latex')


@pytest.mark.parametrize(
    ('template_name', 'document_class', 'title'),
    [
        ('basic_zh.tex', r'\documentclass[UTF8,zihao=-4]{ctexart}', '中文标题'),
        ('basic_en.tex', r'\documentclass[11pt,a4paper]{article}', 'English Title'),
    ],
)
def test_bundled_templates_render(template_name, document_class, title):
    result = _run_filter(
        f'# {title}\n\n## Section\n\n- [ ] pending\n- [x] completed\n',
        template_name=template_name,
    )

    assert result.returncode == 0, result.stderr
    assert document_class in result.stdout
    assert f'\\title{{{title}}}' in result.stdout
    assert '\\section{Section}' in result.stdout
    assert '\\item[\\taskunchecked] pending' in result.stdout
    assert '\\item[\\taskchecked] completed' in result.stdout


def test_pandoc_preserves_standard_markdown_constructs():
    result = _run_filter(r'''# Report

Cost is 50% & $x_1 + \alpha$ with **bold** and `a_b`.

$$
E = mc^2
$$

| Name | Value |
| --- | --- |
| A&B | 20% |

3. First
   - Nested
4. Second

```python
a_b = 1
```
''')

    assert result.returncode == 0, result.stderr
    assert r'50\% \& \(x_1 + \alpha\)' in result.stdout
    assert '\\[\nE = mc^2\n\\]' in result.stdout
    assert r'\textbf{bold}' in result.stdout
    assert r'\texttt{a\_b}' in result.stdout
    assert r'\begin{longtable}' in result.stdout
    assert result.stdout.count(r'p{(\linewidth - 2\tabcolsep) * \real{0.5000}}') == 2
    assert r'A\&B' in result.stdout and r'20\%' in result.stdout
    assert r'\begin{enumerate}' in result.stdout
    assert r'\setcounter{enumi}{2}' in result.stdout
    assert r'\begin{itemize}' in result.stdout
    assert 'Nested' in result.stdout and 'Second' in result.stdout
    assert r'\NormalTok{a\_b }' in result.stdout


def test_markdown_contract_converts_objects_and_internal_references():
    source = normalize_writer_markdown_for_latex(
        '# Document title\n\n'
        '<a id="block-section"></a>\n## Section\n\n'
        '<a id="block-image"></a>\n![Figure 1 Architecture](/tmp/image.png)\n\n'
        '<a id="block-table"></a>\n表1 Results\n| A |\n|---|\n| 1 |\n\n'
        '<a id="block-code"></a>\n代码1 Example\n```python\nprint("ok")\n```\n\n'
        '<a id="block-formula"></a>\n$$\nx+y=z\n$$\n\n'
        'See [**section**](#block-section), [image](#block-image), '
        '[table](#block-table), [code](#block-code), and [formula](#block-formula).\n'
    )
    result = _run_filter(source)

    assert result.returncode == 0, result.stderr
    assert '\\section{Section}\\label{sec:section}' in result.stdout
    assert '\\begin{figure}' in result.stdout
    assert '{assets/image.png}' in result.stdout
    assert '\\caption{Architecture}\\label{fig:image}' in result.stdout
    assert '\\caption{Results}\\label{table:table}' in result.stdout
    assert '\\refstepcounter{codeblock}\\label{code:code}' in result.stdout
    assert '\\label{eq:formula}' in result.stdout
    for label in ('sec:section', 'fig:image', 'table:table', 'code:code', 'eq:formula'):
        assert f'\\writerinternalref{{{label}}}' in result.stdout
    assert '\\textbf{section}' in result.stdout


def test_markdown_contract_renders_numbered_bibliography():
    source = normalize_writer_markdown_for_latex(
        '# Document title\n\n已有研究\\[2]建立在早期工作\\[1]之上。\n\n'
        '## References\n\n\\[1] A & B. First.\n\n\\[2] C. Second.\n'
    )
    result = _run_filter(source, template_name='basic_zh.tex')

    assert result.returncode == 0, result.stderr
    assert '\\cite{ref-002}' in result.stdout
    assert '\\cite{ref-001}' in result.stdout
    assert '\\begin{thebibliography}{9}' in result.stdout
    assert '\\bibitem{ref-001}' in result.stdout
    assert '\\bibitem{ref-002}' in result.stdout
