from typing import List, Optional, Dict, Any
from lazyllm import pipeline, LOG
from lazyllm.tools.data import refine, filter, kbc, embedding, domain_finetune
from lazyllm.tools.data.prompts.domain_finetune import DOMAIN_INSTRUCTION_EN, DOMAIN_PRESETS

DOMAIN_FINETUNE_FEATURES = (
    'normalization',
    'adaptive_normalization',
    'conversation_expand',
    'llm_extraction',
    'deduplication',
    'augmentation',
    'llm_cleaning',
    'output_quality_filter',
)


_CLEANING_STEPS = (
    ('remove_emoji', refine.remove_emoji),
    ('remove_html_url', refine.remove_html_url),
    ('remove_html_entity', refine.remove_html_entity),
    ('remove_extra_spaces', refine.remove_extra_spaces),
)


def build_text_cleaning_pipeline(
    input_key: str = 'content',
    output_key: str = 'cleaned_content',
    remove_emoji: bool = True,
    remove_html_url: bool = True,
    remove_html_entity: bool = True,
    remove_extra_spaces: bool = True,
    enable_llm_cleaning: bool = False,
    llm=None,
    lang: str = 'zh',
) -> pipeline:
    lang = 'zh' if lang in ('zh', 'cn', 'chinese') else 'en'
    flags = (remove_emoji, remove_html_url, remove_html_entity, remove_extra_spaces)
    with pipeline() as ppl:
        for (name, op_factory), flag in zip(_CLEANING_STEPS, flags):
            if flag:
                setattr(ppl, name, op_factory(input_key=input_key))
        if enable_llm_cleaning and llm is not None:
            ppl.llm_clean = kbc.KBCGenerateCleanedTextSingle(
                input_key=input_key,
                llm=llm,
                lang=lang,
            )
            ppl.extract_cleaned = kbc.extract_cleaned_content_single(output_key=input_key)
        ppl.rename = domain_finetune.rename_key(
            input_key=input_key, output_key=output_key, remove_input=True
        )
    return ppl


def build_source_to_content_pipeline(
    source_key: str = 'source',
    output_key: str = 'content',
    mineru_url: str = '',
    mineru_backend: str = 'vlm-vllm-async-engine',
    intermediate_dir: str = 'intermediate',
):
    with pipeline() as ppl:
        ppl.normalize = kbc.FileOrURLNormalizer(
            input_key=source_key,
            intermediate_dir=intermediate_dir,
        )
        ppl.convert_html = kbc.HTMLToMarkdownConverter()
        ppl.convert_pdf = kbc.PDFToMarkdownConverterAPI(
            mineru_url=mineru_url or None,
            mineru_backend=mineru_backend,
        )
        ppl.prepare_load = domain_finetune.prepare_load_path()
        ppl.load_text = kbc.KBCLoadText(input_key='_path_to_load')
        ppl.rename = domain_finetune.rename_key(
            input_key='_text_content',
            output_key=output_key,
            remove_input=True,
        )
    return ppl




_FILTER_BUILDERS = {
    'word_count': lambda c, ik, lang: filter.word_count_filter(
        input_key=ik, min_words=c.get('min_words', 10), max_words=c.get('max_words', 10000), language=lang),
    'char_count': lambda c, ik, lang: filter.char_count_filter(
        input_key=ik, min_chars=c.get('min_chars', 100), max_chars=c.get('max_chars', 100000)),
    'sentence_count': lambda c, ik, lang: filter.sentence_count_filter(
        input_key=ik, min_sentences=c.get('min_sentences', 3), max_sentences=c.get('max_sentences', 1000), language=lang),
    'target_language': lambda c, ik, lang: filter.TargetLanguageFilter(
        input_key=ik, target_language=c.get('target_language', 'zho_Hans'), threshold=c.get('threshold', 0.6)),
    'stop_word': lambda c, ik, lang: filter.StopWordFilter(input_key=ik, max_ratio=c.get('max_ratio', 0.5), language=lang),
    'unique_word': lambda c, ik, lang: filter.unique_word_filter(
        input_key=ik, min_ratio=c.get('min_ratio', 0.1), use_tokenizer=True, language=lang),
    'null_content': lambda c, ik, lang: filter.null_content_filter(input_key=ik),
    'output_min_length': lambda c, ik, lang: domain_finetune.OutputContentFilter(
        input_key=ik,
        min_output_chars=c.get('min_output_chars', 80),
        output_field=c.get('output_field', 'output'),
    ),
    'output_input_ratio': lambda c, ik, lang: domain_finetune.InputOutputRatioFilter(
        input_key=ik,
        min_ratio=c.get('min_ratio', 0.3),
    ),
}


def build_quality_filter_pipeline(
    input_key: str = 'content',
    filters_config: Optional[List[Dict[str, Any]]] = None,
    language: str = 'zh',
):
    if filters_config is None:
        filters_config = [
            {'type': 'word_count', 'min_words': 10, 'max_words': 10000},
            {'type': 'char_count', 'min_chars': 10, 'max_chars': 100000},
            {'type': 'null_content'},
        ]
    with pipeline() as ppl:
        for i, cfg in enumerate(filters_config):
            cfg = dict(cfg)
            ft = cfg.pop('type')
            if ft in _FILTER_BUILDERS:
                setattr(ppl, f'filter_{i}_{ft}', _FILTER_BUILDERS[ft](cfg, input_key, language))
            else:
                LOG.warning(f'Unknown filter type: {ft}, skipping...')
    return ppl



def build_llm_extraction_pipeline(
    input_key: str = 'content',
    output_key: str = 'content',
    llm=None,
    num_samples: int = 3,
    extract_format: str = 'qa',
    lang: str = 'zh',
    max_input_chars: int = 3000,
    keep_original_keys: Optional[List[str]] = None,
    instruction: Optional[str] = None,
    drop_empty_extraction: bool = False,
):
    with pipeline() as ppl:
        ppl.extract = domain_finetune.LLMDataExtractor(
            input_key=input_key,
            output_key='_extracted_samples',
            llm=llm,
            num_samples=num_samples,
            extract_format=extract_format,
            lang=lang,
            max_input_chars=max_input_chars,
            instruction=instruction or 'You are a helpful assistant.',
        )
        ppl.expand = domain_finetune.SampleExpander(
            samples_key='_extracted_samples',
            output_key=output_key,
            keep_original_keys=keep_original_keys or [],
            drop_empty_extraction=drop_empty_extraction,
        )
    return ppl


_FORMATTERS = {
    'alpaca': domain_finetune.DomainFormatAlpaca,
    'sharegpt': domain_finetune.DomainFormatShareGPT,
    'chatml': domain_finetune.DomainFormatChatML,
    'raw': domain_finetune.DomainFormatRaw,
}


def build_domain_formatting_pipeline(
    domain: str = 'general',
    input_key: str = 'content',
    output_key: str = 'formatted_text',
    instruction: Optional[str] = None,
    output_format: str = 'alpaca',
):
    inst = instruction or DOMAIN_INSTRUCTION_EN.get(domain, DOMAIN_INSTRUCTION_EN['general'])
    formatter_cls = _FORMATTERS.get(output_format, domain_finetune.DomainFormatAlpaca)
    with pipeline() as ppl:
        ppl.format = formatter_cls(input_key=input_key, output_key=output_key, instruction=inst)
    return ppl


_AUGMENT_BUILDERS = {
    'query_rewrite': lambda llm, n, lang, ik: embedding.EmbeddingQueryRewrite(
        llm=llm, num_augments=n, lang=lang, input_key=ik),
    'synonym_replace': lambda llm, n, lang, ik: embedding.EmbeddingAdjacentWordSwap(
        num_augments=n, input_key=ik),
}


def build_data_augmentation_pipeline(
    input_key: str = 'content',
    augment_methods: Optional[List[str]] = None,
    llm=None,
    num_augments: int = 2,
    lang: str = 'zh',
):
    methods = augment_methods or ['query_rewrite']
    with pipeline() as ppl:
        for method in methods:
            if method in _AUGMENT_BUILDERS:
                setattr(ppl, method, _AUGMENT_BUILDERS[method](llm, num_augments, lang, input_key))
    return ppl


def build_domain_finetune_pipeline(
    domain: str = 'general',
    input_key: str = 'content',
    output_key: str = 'formatted_text',
    enabled: Optional[Dict[str, bool]] = None,
    options: Optional[Dict[str, Any]] = None,
    filters_config: Optional[List[Dict[str, Any]]] = None,
    normalization_instruction: Optional[str] = None,
    instruction: Optional[str] = None,
    output_format: str = 'alpaca',
    language: str = 'zh',
    cleaned_key: str = 'cleaned_content',
):
    enabled = enabled or {}
    opts = options or {}
    preset = DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS['general'])
    is_zh = language in ('zh', 'cn', 'chinese')
    lang_key = 'instruction_zh' if is_zh else 'instruction_en'
    default_instruction = preset.get(lang_key, preset.get('instruction_en', ''))
    normalization_instruction = normalization_instruction or default_instruction
    instruction = instruction or default_instruction
    if filters_config is None and preset.get('filters') is not None:
        filters_config = preset['filters']

    _filter_text_key = '_filter_text'
    has_normalization = enabled.get('normalization') or enabled.get('adaptive_normalization')

    with pipeline() as ppl:
        if opts.get('merge_context_question'):
            params = dict(opts.get('merge_context_question_params') or {})
            question_key = params.get('question_key', 'question')
            context_key = params.get('context_key', 'context')
            target_key = params.get('target_key', 'question')
            default_context_label = '背景' if is_zh else 'Context'
            default_question_label = '问题' if is_zh else 'Question'
            ppl.merge_context_question = domain_finetune.merge_context_and_question(
                question_key=question_key,
                context_key=context_key,
                target_key=target_key,
                context_label=params.get('context_label', default_context_label),
                question_label=params.get('question_label', default_question_label),
                drop_context=params.get('drop_context', True),
            )

        if enabled.get('conversation_expand'):
            ppl.conversation_expand = domain_finetune.ConversationListExpander(
                list_key=opts.get('expand_list_key', 'data'),
                question_prefix=opts.get('expand_q_prefix', '问：'),
                answer_prefix=opts.get('expand_a_prefix', '答：'),
                min_question_chars=opts.get('expand_min_q_chars', 8),
                min_answer_chars=opts.get('expand_min_a_chars', 50),
            )

        ppl.normalization = domain_finetune.DatasetFormatNormalizer(
            output_key=input_key,
            text_key=_filter_text_key,
            instruction=normalization_instruction,
            field_mapping=opts.get('field_mapping') or {},
            keep_system=True,
        )
        if enabled.get('adaptive_normalization') and opts.get('llm') is not None:
            ppl.normalization_llm_map = domain_finetune.LLMFieldMapper(
                output_key=input_key,
                text_key=_filter_text_key,
                llm=opts.get('llm'),
                lang=language,
            )
        filter_key = _filter_text_key if has_normalization else input_key
        current_key = input_key

        if enabled.get('llm_extraction'):
            ppl.llm_extraction = build_llm_extraction_pipeline(
                input_key=current_key, output_key=current_key,
                llm=opts.get('llm'), num_samples=opts.get('llm_num_samples', 3),
                extract_format=opts.get('llm_extract_format', 'qa'), lang=language,
                max_input_chars=opts.get('llm_max_input_chars', 3000),
                keep_original_keys=opts.get('llm_keep_original_keys'),
                instruction=instruction,
                drop_empty_extraction=opts.get('llm_drop_empty_extraction', False),
            )
            ppl.refresh_filter_text = domain_finetune.extract_content_text(
                input_key=current_key, output_key=_filter_text_key,
            )

        if not has_normalization:
            ppl.cleaning = build_text_cleaning_pipeline(
                input_key=current_key, output_key=cleaned_key,
                remove_emoji=True, remove_html_url=True, remove_html_entity=True, remove_extra_spaces=True,
                enable_llm_cleaning=enabled.get('llm_cleaning'), llm=opts.get('llm'), lang=language,
            )
            current_key, filter_key = cleaned_key, cleaned_key
        else:
            LOG.warning('Normalization is enabled, skipping cleaning step (content is dict, string cleaning will discard all samples).')

        ppl.filtering = build_quality_filter_pipeline(
            input_key=filter_key,
            filters_config=filters_config,
            language=language,
        )

        if enabled.get('output_quality_filter'):
            _oq_filters: List[Dict[str, Any]] = []
            _min_out = opts.get('min_output_chars', 80)
            _min_ratio = opts.get('min_output_input_ratio', 0.3)
            if _min_out > 0:
                _oq_filters.append({'type': 'output_min_length', 'min_output_chars': _min_out})
            if _min_ratio > 0.0:
                _oq_filters.append({'type': 'output_input_ratio', 'min_ratio': _min_ratio})
            ppl.output_quality_filtering = build_quality_filter_pipeline(
                input_key=current_key,
                filters_config=_oq_filters,
                language=language,
            )

        if enabled.get('deduplication'):
            dedup_method = opts.get('dedup_method', 'hash')
            if dedup_method == 'minhash':
                ppl.deduplication = filter.MinHashDeduplicator(
                    input_key=filter_key,
                    threshold=opts.get('minhash_threshold', 0.85),
                    num_perm=opts.get('minhash_num_perm', 128),
                )
            else:
                ppl.deduplication = domain_finetune.HashDeduplicator(input_key=filter_key)

        if enabled.get('augmentation') and opts.get('augment_methods'):
            ppl.augmentation = build_data_augmentation_pipeline(
                input_key=current_key,
                augment_methods=opts.get('augment_methods'),
                llm=opts.get('llm'),
                num_augments=2,
                lang=language,
            )

        format_instruction = instruction if instruction is not None else normalization_instruction
        formatter_cls = _FORMATTERS.get(output_format, domain_finetune.DomainFormatAlpaca)
        ppl.formatting = formatter_cls(
            input_key=current_key,
            output_key=output_key,
            instruction=format_instruction,
        )

    return ppl


def build_train_test_split_pipeline(
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: Optional[str] = None,
):
    with pipeline() as ppl:
        ppl.split = domain_finetune.TrainValTestSplitter(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify_key=stratify_key,
        )
    return ppl

__all__ = [
    'DOMAIN_FINETUNE_FEATURES',
    'build_text_cleaning_pipeline',
    'build_source_to_content_pipeline',
    'build_llm_extraction_pipeline',
    'build_quality_filter_pipeline',
    'build_domain_formatting_pipeline',
    'build_data_augmentation_pipeline',
    'build_domain_finetune_pipeline',
    'build_train_test_split_pipeline',
]
