from typing import Optional
from lazyllm import pipeline
from lazyllm.tools.data import enQA, Pdf2QA, PT_MM


def build_img2qa_pipeline(
    model,
    gen_image_key: str = 'image_path',
    gen_context_key: str = 'context',
    gen_num_qa: int = 5,
    gen_prompt: Optional[str] = None,
    gen_concurrency_mode: str = 'thread',

    scorer_image_key: str = 'image_path',
    scorer_query_key: str = 'query',
    scorer_answer_key: str = 'answer',
    scorer_prompt: Optional[str] = None,
    scorer_concurrency_mode: str = 'thread',

    post_input_key: str = 'qa_pairs',
    filter_input_key: str = 'quality_score',
    filter_threshold: float = 0.9,
):
    with pipeline() as ppl:
        ppl.generator = PT_MM.VQAGenerator(
            vlm=model,
            image_key=gen_image_key,
            context_key=gen_context_key,
            num_qa=gen_num_qa,
            prompt=gen_prompt,
            _concurrency_mode=gen_concurrency_mode,
        )

        ppl.post_process = enQA.post_processor(
            input_key=post_input_key
        )

        ppl.vqascorer = PT_MM.VQAScorer(
            vlm=model,
            image_key=scorer_image_key,
            query_key=scorer_query_key,
            answer_key=scorer_answer_key,
            prompt=scorer_prompt,
            _concurrency_mode=scorer_concurrency_mode,
        )

        ppl.quality_filter = Pdf2QA.multi_features_filter(
            input_key=filter_input_key,
            threshold=filter_threshold,
        )

    return ppl
