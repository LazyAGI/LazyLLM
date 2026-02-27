import os
import shutil
import tempfile

from lazyllm import config, LOG
from lazyllm.tools.data import refine, chunker, filter


class MockModel:
    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self

class TestRefineOperators:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_remove_extra_spaces(self):
        func = refine.remove_extra_spaces(input_key='content')
        test_cases = [
            ('This   has    extra     spaces', 'This has extra spaces'),
            ('  Leading and trailing  ', 'Leading and trailing'),
            ('Normal text', 'Normal text'),
        ]
        inputs = [{'content': text} for text, _ in test_cases]
        expected = [{'content': expected} for _, expected in test_cases]
        res = func(inputs)
        assert {item['content'] for item in res} == {item['content'] for item in expected}

    def test_remove_emoji(self):
        func = refine.remove_emoji(input_key='content')
        test_cases = [
            ('Hello 😊 World 🌍!', 'Hello  World !'),
            ('Python 🐍 is awesome 👍', 'Python  is awesome '),
            ('No emoji here', 'No emoji here'),
        ]
        inputs = [{'content': text} for text, _ in test_cases]
        expected = [{'content': expected} for _, expected in test_cases]
        res = func(inputs)
        assert {item['content'] for item in res} == {item['content'] for item in expected}

    def test_remove_html_url(self):
        func = refine.remove_html_url(input_key='content')
        test_cases = [
            ('Check https://example.com for details', 'Check  for details'),
            ('<div>HTML <b>tags</b></div> removed', 'HTML tags removed'),
            ('Visit http://test.com and <a>click</a>', 'Visit  and click'),
        ]
        inputs = [{'content': text} for text, _ in test_cases]
        expected = [{'content': expected} for _, expected in test_cases]
        res = func(inputs)
        assert {item['content'] for item in res} == {item['content'] for item in expected}

    def test_remove_html_entity(self):
        func = refine.remove_html_entity(input_key='content')
        test_cases = [
            ('Hello&nbsp;World', 'HelloWorld'),
            ('&lt;tag&gt; and &amp; symbol', 'tag and  symbol'),
            ('&quot;quoted&quot; text', 'quoted text'),
        ]
        inputs = [{'content': text} for text, _ in test_cases]
        expected = [{'content': expected} for _, expected in test_cases]
        res = func(inputs)
        assert {item['content'] for item in res} == {item['content'] for item in expected}

    def test_refine_chained_operations(self):
        inputs = [{'content': 'Hello 😊  World  https://example.com &nbsp; <b>Bold</b>'}]
        res = refine.remove_emoji(input_key='content')(inputs)
        res = refine.remove_html_url(input_key='content')(res)
        res = refine.remove_html_entity(input_key='content')(res)
        res = refine.remove_extra_spaces(input_key='content')(res)
        assert len(res) == 1
        assert res[0]['content'] == 'Hello World Bold'


class TestChunkerOperators:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_token_chunker(self):
        func = chunker.TokenChunker(input_key='content', max_tokens=50, min_tokens=10, _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        long_text = '人工智能是计算机科学的一个分支。' * 20
        inputs = [{'content': long_text, 'meta_data': {'source': f'doc_{i}'}} for i in range(100)]
        res = func(inputs)
        assert len(res) > len(inputs)
        for chunk in res[:5]:
            assert 'uid' in chunk
            assert 'content' in chunk
            assert 'meta_data' in chunk
            assert 'index' in chunk['meta_data']
            assert 'total' in chunk['meta_data']
            assert 'length' in chunk['meta_data']
            assert chunk['meta_data']['length'] == len(chunk['content'])


class TestFilterOperators:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_language_filter(self):
        model_path = None
        if config['model_path']:
            model_path = os.path.join(config['model_path'], 'fasttext-language_identification')
            if not os.path.exists(model_path):
                model_path = None
        func = filter.TargetLanguageFilter(
            input_key='content', target_language='zho_Hans', threshold=0.3,
            _concurrency_mode='thread', model_path=model_path)
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('这是一段中文文本。', True),
            ('这是 mixed with English.', False),
            ('This is pure English text.', False),
        ]
        inputs = [
            {'uid': f'{i}', 'content': text}
            for i, (text, _) in enumerate(test_cases * 100)
        ]
        expected = [
            {'uid': f'{i}', 'content': text}
            for i, (text, keep) in enumerate(test_cases * 100) if keep
        ]
        res = func(inputs)
        res_map = {item['uid']: item['content'] for item in res}
        expected_map = {item['uid']: item['content'] for item in expected}
        assert res_map == expected_map

    def test_minhash_deduplicate_filter(self):
        func = filter.MinHashDeduplicator(input_key='content', threshold=0.85)
        test_cases = [
            {'uid': '0', 'content': '这是第一段不同的内容。'},
            {'uid': '1', 'content': '这是第二段完全不同的内容。'},
            {'uid': '2', 'content': '这是第一段不同的内容。'},
            {'uid': '3', 'content': '这是第三段独特的内容。'},
            {'uid': '4', 'content': '这是第二段完全不同的内容。'},
        ]
        res = func(test_cases)
        assert len(res) == 3
        result_uids = {item['uid'] for item in res}
        assert result_uids == {'0', '1', '3'}

    def test_blocklist_filter(self):
        func = filter.WordBlocklistFilter(input_key='content',
                                          blocklist=['敏感', '违禁', 'badword'],
                                          threshold=0, language='zh',
                                          _max_workers=64, _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        assert func._max_workers == 64
        test_cases = [
            ('这是正常的文本内容。', True),
            ('这里包含敏感词的内容。', False),
            ('这里有违禁词和敏感词。', False),
            ('Clean text without issues', True),
            ('Text with badword inside', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 100]
        expected = [{'content': text} for text, keep in test_cases * 100 if keep]
        res = func(inputs)
        res_contents = {item['content'] for item in res}
        expected_contents = {item['content'] for item in expected}
        assert res_contents == expected_contents
        assert len(res) == len(expected)

    def test_word_count_filter(self):
        func = filter.word_count_filter(input_key='content', min_words=5, max_words=20,
                                        language='zh', _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('短文本', False),
            ('这是一段适中长度的中文文本内容。', True),
            ('这是一段非常长的文本内容' * 50, False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 500]
        res = func(inputs)
        for item in res:
            text = item['content']
            count = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
            assert 5 <= count < 20

    def test_colon_end_filter(self):
        func = filter.colon_end_filter(input_key='content', _max_workers=128, _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        assert func._max_workers == 128
        test_cases = [
            ('这是正常结尾。', True),
            ('这是冒号结尾：', False),
            ('This ends with colon:', False),
            ('Normal ending.', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 100]
        res = func(inputs)
        assert len(res) == 200

    def test_sentence_count_filter(self):
        func = filter.sentence_count_filter(input_key='content', min_sentences=2, max_sentences=10,
                                            language='zh', _max_workers=64, _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        assert func._max_workers == 64
        test_cases = [
            ('单句。', False),
            ('第一句。第二句。', True),
            ('第一句。第二句！第三句？', True),
            ('这是第一句。这是第二句。这是第三句。', True),
            ('句子' * 20 + '。', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 100]
        res = func(inputs)
        assert len(res) == 300

    def test_ellipsis_end_filter(self):
        func = filter.ellipsis_end_filter(input_key='content', max_ratio=0.3, _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('第一行。\n第二行。\n第三行。', True),
            ('第一行...\n第二行...\n第三行...', False),
            ('第一行…\n第二行。', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 50]
        res = func(inputs)
        assert len(res) == 50

    def test_null_content_filter(self):
        func = filter.null_content_filter(input_key='content', _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Valid content', True),
            ('', False),
            ('   ', False),
            ('有效内容', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 500]
        expected = [{'content': text} for text, keep in test_cases * 500 if keep]
        res = func(inputs)
        assert sorted(res, key=lambda x: x['content']) == sorted(expected, key=lambda x: x['content'])

    def test_word_length_filter(self):
        func = filter.word_length_filter(input_key='content', min_length=3, max_length=10,
                                         _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('I am ok', False),
            ('This is a normal sentence', True),
            ('Supercalifragilisticexpialidocious', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 500]
        res = func(inputs)
        assert len(res) > 0

    def test_symbol_ratio_filter(self):
        func = filter.SymbolRatioFilter(input_key='content', max_ratio=0.3,
                                        _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Normal text without symbols', True),
            ('Text # with ... some … symbols', True),
            ('### ... … ### ... …', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 200]
        res = func(inputs)
        assert len(res) == 200

    def test_idcard_filter(self):
        func = filter.idcard_filter(input_key='content', threshold=3, _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('这是正常文本', True),
            ('请提供身份证号码和ID number还有证件号', False),
            ('Normal text without ID terms', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 200]
        res = func(inputs)
        assert len(res) == 400

    def test_no_punc_filter(self):
        func = filter.no_punc_filter(input_key='content', max_length_between_punct=20,
                                     language='zh', _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('这是。正常。文本。', True),
            ('这是一段没有标点符号的超长文本' * 10, False),
            ('短文本。', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 200]
        res = func(inputs)
        assert len(res) == 400

    def test_special_char_filter(self):
        func = filter.special_char_filter(input_key='content', _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Normal text 正常文本', True),
            ('Text with \u200b zero width space', False),
            ('Text with \ufffd replacement char', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 200]
        res = func(inputs)
        assert len(res) == 200

    def test_watermark_filter(self):
        func = filter.watermark_filter(input_key='content', _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Normal content without watermark', True),
            ('This document contains Copyright notice', False),
            ('Confidential information inside', False),
            ('版权所有 2024', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 200]
        res = func(inputs)
        assert len(res) == 200

    def test_stop_word_filter(self):
        func = filter.StopWordFilter(input_key='content', max_ratio=0.5, use_tokenizer=True,
                                     language='zh', _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('这是一段包含实际内容的正常文本。', True),
            ('的了吗呢吧啊', False),
            ('人工智能技术发展迅速。', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 20]
        res = func(inputs)
        assert len(res) == 40

    def test_curly_bracket_filter(self):
        func = filter.curly_bracket_filter(input_key='content', max_ratio=0.08,
                                           _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Normal text without brackets', True),
            ('Text with {one} bracket', True),
            ('{{{{{' * 10, False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 100]
        res = func(inputs)
        assert len(res) == 100

    def test_capital_word_filter(self):
        func = filter.CapitalWordFilter(input_key='content', max_ratio=0.5,
                                        _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Normal text with Some Capitals', True),
            ('MOSTLY UPPERCASE TEXT HERE', False),
            ('mixed CaSe TeXt', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 50]
        res = func(inputs)
        assert len(res) == 100

    def test_lorem_ipsum_filter(self):
        func = filter.lorem_ipsum_filter(input_key='content', max_ratio=3e-8,
                                         _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('This is real content', True),
            ('Lorem ipsum dolor sit amet', False),
            ('占位符文本', False),
            ('Normal Chinese text 正常文本', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 20]
        res = func(inputs)
        assert len(res) == 40

    def test_unique_word_filter(self):
        func = filter.unique_word_filter(input_key='content', min_ratio=0.3, use_tokenizer=True,
                                         language='zh', _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('这是一段包含多个不同词汇的文本内容。', True),
            ('重复重复重复重复重复', False),
            ('人工智能和机器学习技术。', True),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 30]
        res = func(inputs)
        assert len(res) == 60

    def test_char_count_filter(self):
        func = filter.char_count_filter(input_key='content', min_chars=10, max_chars=100,
                                        _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('短', False),
            ('这是一段中等长度的文本内容。', True),
            ('这是一段超长文本' * 50, False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 50]
        res = func(inputs)
        assert len(res) == 50

    def test_bullet_point_filter(self):
        func = filter.bullet_point_filter(input_key='content', max_ratio=0.5,
                                          _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Normal paragraph text', True),
            ('• Item 1\n• Item 2\n• Item 3', False),
            ('• A\n• B\n• C\n• D\n• E\n• F', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 50]
        res = func(inputs)
        assert len(res) == 50

    def test_javascript_filter(self):
        func = filter.javascript_filter(input_key='content', min_non_script_lines=2,
                                        _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        test_cases = [
            ('Short normal text', True),
            ('Intro paragraph here.\nSecond line of content.\nThird line.\nFourth line.\nAll normal.',
             True),
            ('function() { return 1; }\nconst x = 1;\nvar y = 2;\nlet z = 3;', False),
        ]
        inputs = [{'content': text} for text, _ in test_cases * 50]
        res = func(inputs)
        assert len(res) == 100
