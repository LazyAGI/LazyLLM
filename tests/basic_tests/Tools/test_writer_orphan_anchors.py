import shutil
import subprocess
import unittest
from pathlib import Path


class TestWriterOrphanAnchors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pandoc = shutil.which('pandoc')
        if cls.pandoc is None:
            raise unittest.SkipTest('Pandoc is not installed.')
        cls.filter = Path(__file__).resolve().parents[3] / 'lazyllm/tools/writer/templates/latex/filters/writer.lua'

    def convert(self, source):
        return subprocess.run(
            [self.pandoc, '--from=gfm-yaml_metadata_block', '--to=latex', '--lua-filter', str(self.filter)],
            input=source, capture_output=True, text=True, check=False,
        )

    def test_orphan_before_valid_target(self):
        for attributes in ('', ' data-kind="figure" data-caption="Missing caption"'):
            with self.subTest(attributes=attributes):
                result = self.convert(
                    '# Document\n\n[Missing](#block-missing) and [Section](#block-section).\n\n'
                    f'<a id="block-missing"{attributes}></a>\n\n'
                    '<a id="block-section" data-kind="section" data-caption="Section"></a>\n\n'
                    '## Section\n\nBody text.\n'
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Missing and Section', result.stdout)
                self.assertIn('Body text.', result.stdout)
                self.assertIn(r'\label{sec:section}', result.stdout)
                self.assertIn(r'\writerinternalref{sec:section}', result.stdout)
                self.assertNotIn('fig:missing', result.stdout)
                self.assertNotIn('sec:missing', result.stdout)
                self.assertIn('anchor has no target: block-missing', result.stderr)

    def test_consecutive_and_trailing_orphans(self):
        result = self.convert(
            '# Document\n\n[First](#block-first) and [Last](#block-last).\n\n'
            '<a id="block-first"></a>\n\n'
            '<a id="block-last" data-kind="figure" data-caption="Missing"></a>\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('First and Last.', result.stdout)
        self.assertNotIn(r'\writerinternalref', result.stdout)
        self.assertIn('anchor has no target: block-first', result.stderr)
        self.assertIn('anchor has no target: block-last', result.stderr)

    def test_valid_image_and_code_example_survive(self):
        result = self.convert(
            '# Document\n\n<a id="block-missing"></a>\n\n'
            '<a id="block-image" data-kind="figure" data-caption="Diagram"></a>\n\n'
            '![Diagram](/tmp/image.png)\n\n[Image](#block-image).\n\n'
            '```\n<a id="block-code-example"></a>\n```\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(r'\label{fig:image}', result.stdout)
        self.assertIn(r'\writerinternalref{fig:image}', result.stdout)
        self.assertIn('assets/image.png', result.stdout)
        self.assertIn('<a id="block-code-example"></a>', result.stdout)

    def test_duplicate_and_kind_conflict_still_fail(self):
        for source, diagnostic in (
            ('<a id="block-same"></a>\n## One\n\n<a id="block-same"></a>\n## Two\n', 'duplicate anchor'),
            ('<a id="block-wrong" data-kind="figure"></a>\n## Section\n', 'kind does not match target'),
        ):
            with self.subTest(diagnostic=diagnostic):
                result = self.convert('# Document\n\n' + source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_portable_conversion_with_orphan_anchor(self):
        from lazyllm.tools.writer.provider import WriterProviderBase

        source = (
            '# Document\n\n[Missing](#block-missing) and [Section](#block-section).\n\n'
            '<a id="block-missing"></a>\n\n'
            '<a id="block-section"></a>\n## Section\n\nBody text.\n'
        )
        result = WriterProviderBase.convert_common_document(source, output_format='latex')
        self.assertEqual(result.format, 'latex')
        self.assertIn(r'\documentclass', result.content)
        self.assertIn('Missing and Section', result.content)
        self.assertIn(r'\writerinternalref{sec:section}', result.content)
        self.assertIn('Body text.', result.content)


if __name__ == '__main__':
    unittest.main()
