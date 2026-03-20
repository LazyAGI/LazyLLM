import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data import domain_pretrain


class TestDomainPretrainOperators:
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

    # ---------- PretrainFieldNormalizer ----------

    def test_pretrain_field_normalizer_existing_content(self):
        op = domain_pretrain.PretrainFieldNormalizer(
            content_key='content', _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': '  already  '}])[0]
        assert r['content'] == '  already  '

    def test_pretrain_field_normalizer_concat(self):
        op = domain_pretrain.PretrainFieldNormalizer(
            content_key='content',
            concat_fields=['title', 'body'],
            concat_separator=' | ',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'title': ' T ', 'body': 'B'}])[0]
        assert r['content'] == 'T | B'

    def test_pretrain_field_normalizer_fallback(self):
        op = domain_pretrain.PretrainFieldNormalizer(
            content_key='content',
            fallback_fields=['text'],
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'text': 'fallback body'}])[0]
        assert r['content'] == 'fallback body'

    def test_pretrain_field_normalizer_field_mapping(self):
        op = domain_pretrain.PretrainFieldNormalizer(
            content_key='content',
            field_mapping={'raw_body': 'content'},
            drop_original=False,
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'raw_body': 'mapped'}])[0]
        assert r['content'] == 'mapped' and r['raw_body'] == 'mapped'

    def test_pretrain_field_normalizer_forward_none(self):
        op = domain_pretrain.PretrainFieldNormalizer(
            content_key='content',
            fallback_fields=[],
            concat_fields=None,
            _save_data=False, _concurrency_mode='single',
        )
        assert op.forward({'meta': 1}) is None

    # ---------- DomainKeywordFilter ----------

    def test_domain_keyword_filter_no_keywords_pass_through(self):
        op = domain_pretrain.DomainKeywordFilter(
            input_key='content', keywords=[],
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'anything'}])[0]
        assert r['content'] == 'anything' and '_keyword_hits' not in r

    def test_domain_keyword_filter_hit(self):
        op = domain_pretrain.DomainKeywordFilter(
            input_key='content',
            keywords=['python'],
            min_keyword_hits=1,
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'I love Python language'}])[0]
        assert r['_keyword_hits'] >= 1 and '_keyword_types' in r

    def test_domain_keyword_filter_forward_none(self):
        op = domain_pretrain.DomainKeywordFilter(
            input_key='content',
            keywords=['missing'],
            _save_data=False, _concurrency_mode='single',
        )
        assert op.forward({'content': 'no keyword here'}) is None

    def test_domain_keyword_filter_density_mode(self):
        op = domain_pretrain.DomainKeywordFilter(
            input_key='content',
            keywords=['a'],
            mode='density',
            min_keyword_density=0.01,
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'a' * 50}])[0]
        assert r['_keyword_hits'] >= 1

    # ---------- DomainRelevanceScorer ----------

    def test_domain_relevance_scorer_no_keywords(self):
        op = domain_pretrain.DomainRelevanceScorer(
            input_key='content', keywords=[], min_score=0.5,
            language='en',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'short'}])[0]
        assert r['_domain_relevance_score'] == 1.0

    def test_domain_relevance_scorer_en_pass(self):
        op = domain_pretrain.DomainRelevanceScorer(
            input_key='content',
            keywords=['alpha'],
            min_score=0.001,
            language='en',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'alpha beta alpha gamma'}])[0]
        assert r['_domain_relevance_score'] > 0

    def test_domain_relevance_scorer_forward_none(self):
        op = domain_pretrain.DomainRelevanceScorer(
            input_key='content',
            keywords=['zzz'],
            min_score=10.0,
            language='en',
            _save_data=False, _concurrency_mode='single',
        )
        assert op.forward({'content': 'alpha beta'}) is None

    # ---------- NGramRepetitionFilter ----------

    def test_ngram_repetition_filter_low_repetition_pass(self):
        op = domain_pretrain.NGramRepetitionFilter(
            input_key='content',
            n=3,
            max_repetition_ratio=0.9,
            language='en',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'the quick brown fox jumps'}])[0]
        assert '_ngram_repetition_ratio' in r

    def test_ngram_repetition_filter_high_repetition_drop(self):
        op = domain_pretrain.NGramRepetitionFilter(
            input_key='content',
            n=2,
            max_repetition_ratio=0.01,
            language='en',
            _save_data=False, _concurrency_mode='single',
        )
        spam = ' '.join(['foo bar'] * 30)
        assert op.forward({'content': spam}) is None

    def test_ngram_repetition_filter_short_text(self):
        op = domain_pretrain.NGramRepetitionFilter(
            input_key='content', n=20, language='en',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'short'}])[0]
        assert r['_ngram_repetition_ratio'] == 0.0

    def test_sensitive_info_cleaner(self):
        op = domain_pretrain.sensitive_info_cleaner(
            input_key='content', _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'call 13812345678 or a@b.co ok'}])[0]
        assert '13812345678' not in r['content']
        assert 'a@b.co' not in r['content']
        assert '[REDACTED]' in r['content']

    def test_text_normalizer(self):
        op = domain_pretrain.text_normalizer(
            input_key='body', _save_data=False, _concurrency_mode='single',
        )
        r = op([{'body': '  a\t\tb  \n\n\n  c  '}])[0]
        assert '\t' not in r['body']
        assert r['body'].startswith('a')

    # ---------- DocumentLanguageFilter ----------

    def test_document_language_filter_zh_pass(self):
        op = domain_pretrain.DocumentLanguageFilter(
            input_key='content',
            target_language='zh',
            min_target_ratio=0.2,
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': '这是一段用于测试的中文正文内容，足够长度。'}])[0]
        assert '_language_ratio' in r

    def test_document_language_filter_en_pass(self):
        op = domain_pretrain.DocumentLanguageFilter(
            input_key='content',
            target_language='en',
            min_target_ratio=0.5,
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'hello world this is english text'}])[0]
        assert '_language_ratio' in r

    def test_document_language_filter_zh_fail_english_only(self):
        op = domain_pretrain.DocumentLanguageFilter(
            input_key='content',
            target_language='zh',
            min_target_ratio=0.9,
            _save_data=False, _concurrency_mode='single',
        )
        assert op.forward({'content': 'only ascii letters here'}) is None

    def test_document_language_filter_unknown_target_skips(self):
        op = domain_pretrain.DocumentLanguageFilter(
            input_key='content',
            target_language='fr',
            min_target_ratio=0.99,
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'anything'}])[0]
        assert r['content'] == 'anything' and '_language_ratio' not in r
