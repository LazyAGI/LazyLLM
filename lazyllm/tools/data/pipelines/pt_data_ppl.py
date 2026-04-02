from lazyllm import pipeline
from lazyllm.tools.data import filter
from lazyllm.tools.data import pt
from lazyllm.tools.data import pt_mm
from lazyllm.tools.data import refine
from lazyllm.tools.data.operators.token_chunker import TokenChunker


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
        ppl.char_count_filter = filter.char_count_filter(
            input_key=content_key, min_chars=min_chars, max_chars=max_chars
        )
        ppl.word_count_filter = filter.word_count_filter(
            input_key=content_key,
            min_words=min_words,
            max_words=max_words,
            language=language,
        )
        ppl.sentence_count_filter = filter.sentence_count_filter(
            input_key=content_key, language=language
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
        ppl.remove_html_url = refine.remove_html_url(input_key=content_key)
        ppl.remove_html_entity = refine.remove_html_entity(input_key=content_key)
        ppl.remove_emoji = refine.remove_emoji(input_key=content_key)
        ppl.remove_extra_spaces = refine.remove_extra_spaces(input_key=content_key)
        ppl.token_chunker = TokenChunker(
            input_key=content_key,
            max_tokens=max_tokens,
            min_tokens=min_tokens
        )
    return ppl


def build_phi4_pt_pipeline(
        context_key='context',
        image_key=None,
        llm=None,
        num_qa=5):
    with pipeline() as ppl:
        ppl.context_qual_filter = pt.ContextQualFilter(
            llm=llm.share(),
            context_key=context_key,
            image_key=image_key
        )
        ppl.phi4_qa_generator = pt.Phi4QAGenerator(
            llm=llm.share(),
            context_key=context_key,
            image_key=image_key,
            num_qa=num_qa
        )
    return ppl


def build_mm_pt_pipeline(
        image_key='image_path',
        text_key='text',
        vlm=None,
        min_width=256,
        min_height=256,
        max_side=1024,
        relevance_threshold=0.6,
        use_dedup=True):
    with pipeline() as ppl:
        ppl.integrity_check = pt_mm.integrity_check(image_key=image_key)
        ppl.resolution_filter = pt_mm.resolution_filter(
            image_key=image_key,
            min_width=min_width,
            min_height=min_height
        )
        ppl.resolution_resize = pt_mm.resolution_resize(
            image_key=image_key,
            max_side=max_side
        )
        if use_dedup:
            ppl.image_dedup = pt_mm.ImageDedup(image_key=image_key)
        if vlm is not None:
            ppl.text_relevance_filter = pt_mm.TextRelevanceFilter(
                vlm=vlm.share(),
                image_key=image_key,
                text_key=text_key,
                threshold=relevance_threshold
            )
    return ppl


def build_structured_data_pipeline(llm, input_key='text', output_key='parsed', prompt=None):
    with pipeline() as ppl:
        ppl.text2json = pt.Text2Json(
            llm,
            input_key=input_key,
            output_key=output_key,
            prompt=prompt,
        )
    return ppl


def build_long_context_pipeline(llm, context_key='context', question_key='question',
                                answer_key='answer', expanded_key='expanded_context',
                                long_context_key='long_context',
                                expansion_prompt=None, num_distractors=3,
                                passage_sep='\n\n', seed=None,
                                expansion_concurrency_mode='thread',
                                reconstruction_concurrency_mode='thread'):
    with pipeline() as ppl:
        ppl.context_expansion = pt.ContextExpansion(
            llm=llm,
            context_key=context_key,
            question_key=question_key,
            answer_key=answer_key,
            expanded_key=expanded_key,
            prompt=expansion_prompt,
            _concurrency_mode=expansion_concurrency_mode,
        )
        ppl.context_reconstruction = pt.ContextReconstruction(
            context_key=expanded_key,
            question_key=question_key,
            answer_key=answer_key,
            long_context_key=long_context_key,
            num_distractors=num_distractors,
            passage_sep=passage_sep,
            seed=seed,
            _concurrency_mode=reconstruction_concurrency_mode,
        )
    return ppl
