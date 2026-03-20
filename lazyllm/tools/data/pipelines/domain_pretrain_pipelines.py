from typing import List, Optional, Dict, Any
from lazyllm import pipeline
from lazyllm.tools.data import filter, refine, domain_pretrain
from lazyllm.tools.data.operators.token_chunker import TokenChunker
from lazyllm.tools.data.prompts.domain_finetune import DOMAIN_PRESETS

DOMAIN_PRETRAIN_FEATURES = (
    'field_normalization', 
    'text_normalization',
    'sensitive_info_cleaning',
    'language_filter',
    'domain_keyword_filter',
    'domain_relevance_scorer',
    'ngram_repetition_filter',
)


def build_text_pt_pipeline(
        content_key='content',
        language='zh',
        min_chars=100,
        max_chars=100000,
        min_words=10,
        max_words=10000,
        max_tokens=1024,
        min_tokens=200):
    with pipeline() as ppl:
        ppl.null_content_filter = filter.null_content_filter(input_key=content_key)
        ppl.remove_html_url = refine.remove_html_url(input_key=content_key)
        ppl.remove_html_entity = refine.remove_html_entity(input_key=content_key)
        ppl.remove_emoji = refine.remove_emoji(input_key=content_key)
        ppl.remove_extra_spaces = refine.remove_extra_spaces(input_key=content_key)
        ppl.char_count_filter = filter.char_count_filter(
            input_key=content_key, min_chars=min_chars, max_chars=max_chars
        )
        ppl.word_count_filter = filter.word_count_filter(
            input_key=content_key, min_words=min_words, max_words=max_words, language=language
        )

        ppl.sentence_count_filter = filter.sentence_count_filter(
            input_key=content_key, language=language, min_sentences=1, max_sentences=100000
        )
        ppl.special_char_filter = filter.special_char_filter(input_key=content_key)
        ppl.watermark_filter = filter.watermark_filter(input_key=content_key)
        ppl.idcard_filter = filter.idcard_filter(input_key=content_key)
        ppl.javascript_filter = filter.javascript_filter(input_key=content_key)
        ppl.lorem_ipsum_filter = filter.lorem_ipsum_filter(input_key=content_key)
        ppl.colon_end_filter = filter.colon_end_filter(input_key=content_key)
        ppl.ellipsis_end_filter = filter.ellipsis_end_filter(input_key=content_key)
        ppl.no_punc_filter = filter.no_punc_filter(input_key=content_key, language=language)
        ppl.curly_bracket_filter = filter.curly_bracket_filter(input_key=content_key)
        ppl.bullet_point_filter = filter.bullet_point_filter(input_key=content_key)
        ppl.symbol_ratio_filter = filter.SymbolRatioFilter(input_key=content_key)
        ppl.stop_word_filter = filter.StopWordFilter(
            input_key=content_key, language=language
        )
        ppl.unique_word_filter = filter.unique_word_filter(
            input_key=content_key, language=language
        )
        ppl.word_blocklist_filter = filter.WordBlocklistFilter(
            input_key=content_key, language=language
        )
        ppl.minhash_deduplicator = filter.MinHashDeduplicator(input_key=content_key)
        ppl.token_chunker = TokenChunker(
            input_key=content_key,
            max_tokens=max_tokens,
            min_tokens=min_tokens
        )

    return ppl


def build_domain_pretrain_pipeline(
    domain: str = 'general',
    content_key: str = 'content',
    language: str = 'zh',
    enabled: Optional[Dict[str, bool]] = None,
    options: Optional[Dict[str, Any]] = None,
    domain_keywords: Optional[List[str]] = None,
):

    preset = DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS['general'])
    opts = options or {}

    default_enabled = {
        'field_normalization': True,
        'text_normalization': True,
        'sensitive_info_cleaning': True,
        'language_filter': False,
        'domain_keyword_filter': False,
        'domain_relevance_scorer': False,
        'ngram_repetition_filter': True,
    }

    dk_flag = opts.get('enable_domain_keyword_filter')
    if dk_flag is not None:
        default_enabled['domain_keyword_filter'] = bool(dk_flag)

    if enabled is not None:
        default_enabled.update(enabled)
    enabled = default_enabled

    if domain_keywords is None:
        domain_keywords = list(preset.get('pretrain_keywords', []))

    with pipeline() as ppl:

        ppl.field_normalization = domain_pretrain.PretrainFieldNormalizer(
            content_key=content_key,
            field_mapping=opts.get('field_mapping') or {},
            concat_fields=opts.get('concat_fields'),
            concat_separator=opts.get('concat_separator', '\n\n'),
            fallback_fields=opts.get('fallback_fields'),
            drop_original=opts.get('drop_original_fields', False),
        )

        
        ppl.text_normalizer = domain_pretrain.text_normalizer(
            input_key=content_key,
            unicode_form='NFKC',
            fix_encoding=True,
            normalize_whitespace=True,
        )

        ppl.sensitive_cleaner = domain_pretrain.sensitive_info_cleaner(
            input_key=content_key,
            remove_phone=opts.get('remove_phone', True),
            remove_email=opts.get('remove_email', True),
            remove_ip=opts.get('remove_ip', True),
            remove_bank_card=opts.get('remove_bank_card', True),
            replacement=opts.get('sensitive_replacement', '[REDACTED]'),
        )

        if enabled.get('language_filter'):
            ppl.language_filter = domain_pretrain.DocumentLanguageFilter(
                input_key=content_key,
                target_language=language,
                min_target_ratio=opts.get('min_language_ratio', 0.3),
            )

        if enabled.get('domain_keyword_filter') and domain_keywords:
            ppl.domain_keyword_filter = domain_pretrain.DomainKeywordFilter(
                input_key=content_key,
                keywords=domain_keywords,
                mode=opts.get('keyword_mode', 'density'),
                min_keyword_hits=opts.get('min_keyword_hits', 1),
                min_keyword_density=opts.get('min_keyword_density', 0.001),
            )

        if enabled.get('domain_relevance_scorer') and domain_keywords:
            ppl.domain_relevance = domain_pretrain.DomainRelevanceScorer(
                input_key=content_key,
                keywords=domain_keywords,
                keyword_weights=opts.get('keyword_weights', {}),
                min_score=opts.get('min_relevance_score', 0.1),
                language=language,
            )


        ppl.ngram_filter = domain_pretrain.NGramRepetitionFilter(
            input_key=content_key,
            n=opts.get('ngram_n', 10),
            max_repetition_ratio=opts.get('max_repetition_ratio', 0.3),
            language=language,
        )

    return ppl


def build_text_pt_plus_domain_pretrain_pipeline(
    domain: str = 'general',
    content_key: str = 'content',
    language: str = 'zh',
    enabled: Optional[Dict[str, bool]] = None,
    options: Optional[Dict[str, Any]] = None,
    domain_keywords: Optional[List[str]] = None,
):

    opts = options or {}
    lang = (language or 'zh').lower()
    if lang in ('zh', 'cn', 'chinese'):
        def_min_chars, def_min_words, def_min_tokens = 50, 20, 128
    else:
        def_min_chars, def_min_words, def_min_tokens = 200, 50, 200

    domain_enhance_ppl = build_domain_pretrain_pipeline(
        domain=domain,
        content_key=content_key,
        language=language,
        enabled=enabled,
        options=options,
        domain_keywords=domain_keywords,
    )

    text_pt_ppl = build_text_pt_pipeline(
        content_key=content_key,
        language=language,
        min_chars=opts.get('min_chars', def_min_chars),
        max_chars=opts.get('max_chars', 500000),
        min_words=opts.get('min_words', def_min_words),
        max_words=opts.get('max_words', 50000),
        max_tokens=opts.get('max_tokens', 1024),
        min_tokens=opts.get('min_tokens', def_min_tokens),
    )

    with pipeline() as ppl:
        ppl.domain_enhance = domain_enhance_ppl
        ppl.text_pt = text_pt_ppl

    return ppl


__all__ = [
    'DOMAIN_PRETRAIN_FEATURES',
    'build_text_pt_pipeline',
    'build_domain_pretrain_pipeline',
    'build_text_pt_plus_domain_pretrain_pipeline',
]
