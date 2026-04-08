import pytest
from unittest.mock import MagicMock, patch

from lazyllm.module import LLMBase
from lazyllm.tools.rag.query_enh_ac import (
    QueryEnhACProcessor,
    _LLMFilter,
    _BERTFilter,
    _default_llm_prompt,
)


def _mock_llm_discriminator(*call_returns):
    model = MagicMock(spec=LLMBase)
    terminal = MagicMock()
    if len(call_returns) == 1:
        only = call_returns[0]
        if isinstance(only, list) and only and all(isinstance(x, bool) for x in only):
            terminal.return_value = only
        elif isinstance(only, list):
            terminal.side_effect = only
        else:
            terminal.return_value = only
    else:
        terminal.side_effect = list(call_returns)
    model.share.return_value.prompt.return_value.formatter.return_value = terminal
    return model, terminal


class TestQueryEnhACProcessor(object):

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_invalid_prompt_lang_raises(self):
        with pytest.raises(ValueError, match='prompt_lang'):
            QueryEnhACProcessor(data_source=[], discriminator=None, prompt_lang='fr')

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_invalid_discriminator_type_raises(self):
        with pytest.raises(TypeError, match='Unsupported discriminator'):
            QueryEnhACProcessor(data_source=[], discriminator=object())

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_empty_vocab_returns_query_unchanged(self):
        proc = QueryEnhACProcessor(data_source=[], discriminator=None)
        assert proc('任意查询文本') == '任意查询文本'
        assert proc(['a', 'b']) == ['a', 'b']

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_discriminator_none_logs_and_returns_original_when_ac_matches(self):
        vocab = [
            {'cluster_id': 'cx', 'word': 'lazyllmacpodvocabtoken'},
            {'cluster_id': 'cx', 'word': 'alias_ac'},
        ]
        proc = QueryEnhACProcessor(data_source=vocab, discriminator=None)
        q = '请用lazyllmacpodvocabtoken完成检索'
        with patch('lazyllm.tools.rag.query_enh_ac.LOG') as mock_log:
            out = proc(q)
        assert out == q
        mock_log.warning.assert_called()
        wargs = mock_log.warning.call_args[0]
        assert wargs and 'discriminator is None' in str(wargs[0])

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_llm_synonym_enhancement(self):
        vocab = [
            {'cluster_id': 'cx', 'word': 'fooword'},
            {'cluster_id': 'cx', 'word': 'alias_ac'},
        ]
        model, _ = _mock_llm_discriminator([True])
        proc = QueryEnhACProcessor(data_source=vocab, discriminator=model)
        out = proc('fooword')
        assert 'fooword' in out
        assert 'alias_ac' in out

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_get_matches_aligned_with_ac_jieba_shape(self):
        vocab = [
            {'cluster_id': 'cx', 'word': 'fooword'},
            {'cluster_id': 'cx', 'word': 'alias_ac'},
        ]
        model, _ = _mock_llm_discriminator([True])
        proc = QueryEnhACProcessor(data_source=vocab, discriminator=model)
        ms = proc.get_matches('fooword')
        assert len(ms) == 1
        assert ms[0]['word'] == 'fooword'
        assert ms[0]['cluster_id'] == 'cx'
        assert set(ms[0]['cluster_words']) == {'fooword', 'alias_ac'}

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_update_data_source_rebuilds_automaton(self):
        token = 'updqwxyztoken'
        model, _ = _mock_llm_discriminator([True])
        proc = QueryEnhACProcessor(data_source=[], discriminator=model)
        assert proc(f'前缀{token}后缀') == f'前缀{token}后缀'
        proc.update_data_source([
            {'cluster_id': 'u', 'word': token},
            {'cluster_id': 'u', 'word': 'syn_u'},
        ])
        out = proc(f'前缀{token}后缀')
        assert token in out
        assert 'syn_u' in out

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_callable_data_source(self):
        rows = [{'cluster_id': 'k', 'word': 'callabletok'}]

        def ds():
            return list(rows)

        token = 'callabletok'
        model, _ = _mock_llm_discriminator([True])
        proc = QueryEnhACProcessor(data_source=ds, discriminator=model)
        out = proc(f'使用{token}测试')
        assert token in out
        rows.append({'cluster_id': 'k', 'word': 'callable_alias'})
        proc.update_data_source(ds)
        out2 = proc(f'使用{token}测试')
        assert 'callable_alias' in out2

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_custom_cluster_and_word_keys(self):
        token = 'keytokabc'
        vocab = [{'cid': 1, 'w': token}, {'cid': 1, 'w': 'alt_w'}]
        model, _ = _mock_llm_discriminator([True])
        proc = QueryEnhACProcessor(
            data_source=vocab,
            discriminator=model,
            cluster_key='cid',
            word_key='w',
        )
        out = proc(f'X{token}Y')
        assert token in out and 'alt_w' in out

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_prompt_lang_en_init_with_llm(self):
        model, _ = _mock_llm_discriminator([True])
        proc = QueryEnhACProcessor(
            data_source=[],
            discriminator=model,
            prompt_lang='en',
        )
        assert proc('q') == 'q'

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_llm_invalid_output_returns_original_query(self):
        vocab = [
            {'cluster_id': 'c', 'word': 'hi'},
            {'cluster_id': 'c', 'word': 'hello'},
        ]
        model, terminal = _mock_llm_discriminator('not-a-list')
        proc = QueryEnhACProcessor(data_source=vocab, discriminator=model, max_retries=2)
        with patch('lazyllm.tools.rag.query_enh_ac.LOG') as mock_log:
            out = proc('hi')
        assert out == 'hi'
        assert terminal.call_count == 2
        warn_blob = ' '.join(
            str(a) for call in mock_log.warning.call_args_list for a in call.args
        )
        assert '_LLMFilter' in warn_blob


class TestQueryEnhACPrompts(object):

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_default_llm_prompt_zh_and_en_differ(self):
        p_zh = _default_llm_prompt('zh')
        p_en = _default_llm_prompt('en')
        assert p_zh is not None and p_en is not None
        assert p_zh != p_en


class TestLLMFilter(object):

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_preprocess_candidate_labels_zh_vs_en(self):
        def make_filter(lang):
            model = MagicMock(spec=LLMBase)
            model.share.return_value.prompt.return_value.formatter.return_value = MagicMock()
            return _LLMFilter(model=model, prompt_lang=lang)

        f_zh = make_filter('zh')
        d_zh = f_zh._preprocess('什么是民法？', [
            {'word': '民法', 'start': 3, 'end': 4},
        ])
        assert '匹配词="' in d_zh['candidates_text']
        assert '上下文="' in d_zh['candidates_text']

        f_en = make_filter('en')
        d_en = f_en._preprocess('什么是民法？', [
            {'word': '民法', 'start': 3, 'end': 4},
        ])
        assert 'matched_word="' in d_en['candidates_text']
        assert 'context="' in d_en['candidates_text']

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_call_empty_matches_short_circuit(self):
        model = MagicMock(spec=LLMBase)
        model.share.return_value.prompt.return_value.formatter.return_value = MagicMock()
        f = _LLMFilter(model=model)
        assert f('q', []) == []

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_call_failure_returns_empty_list(self):
        model = MagicMock(spec=LLMBase)
        terminal = MagicMock(return_value=None)
        model.share.return_value.prompt.return_value.formatter.return_value = terminal
        f = _LLMFilter(model=model, max_retries=2)
        matches = [{'word': 'a', 'start': 0, 'end': 0}]
        out = f('query', matches)
        assert out == []
        assert terminal.call_count == 2


def _mock_bert_module_chain(*call_returns):
    terminal = MagicMock()
    if len(call_returns) == 1:
        only = call_returns[0]
        if isinstance(only, list):
            terminal.side_effect = only
        elif isinstance(only, BaseException):
            terminal.side_effect = only
        else:
            terminal.return_value = only
    else:
        terminal.side_effect = list(call_returns)
    model = MagicMock()
    model.share.return_value = terminal
    return model, terminal


class TestBERTFilter(object):

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_call_empty_matches(self):
        model, _ = _mock_bert_module_chain('{}')
        f = _BERTFilter(model=model)
        assert f('q', []) == []

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_threshold_keeps_or_drops(self):
        hi = '{"probs": [0.2, 0.8], "predicted_label": 1}'
        lo = '{"probs": [0.8, 0.2], "predicted_label": 0}'
        model, terminal = _mock_bert_module_chain(hi, lo)
        f = _BERTFilter(model=model, threshold=0.5)
        matches = [
            {'word': 'a', 'start': 0, 'end': 0},
            {'word': 'b', 'start': 1, 'end': 1},
        ]
        out = f('querytext', matches)
        assert len(out) == 1 and out[0]['word'] == 'a'
        assert terminal.call_count == 2

    @pytest.mark.run_on_change('lazyllm/tools/rag/query_enh_ac.py')
    def test_inference_failure_drops_match_after_retries(self):
        model, terminal = _mock_bert_module_chain(ValueError('bad'), ValueError('bad'))
        f = _BERTFilter(model=model, max_retries=2)
        matches = [{'word': 'x', 'start': 0, 'end': 0}]
        with patch('lazyllm.tools.rag.query_enh_ac.LOG'):
            out = f('qq', matches)
        assert out == []
        assert terminal.call_count == 2
