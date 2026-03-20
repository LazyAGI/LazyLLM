from lazyllm import pipeline
from lazyllm.tools.data import Pdf2QA, Text2qa


def build_img2qa_pipeline(
    model,
    gen_prompt=None,
    score_prompt=None,
    image_key='image',
    context_key='objects',
    query_key='query',
    answer_key='answer',
    score_key='score',
    reference_key='reference',
    filter_threshold=1,
    img_resize=False,
    size=(336, 336),
    to_chat=False
):
    with pipeline() as ppl:
        if img_resize:
            ppl.resize = Pdf2QA.resize_image_inplace(
                image_key=image_key,
                size=size
            )

        ppl.generator = Pdf2QA.ImageToVQA(
            image_key=image_key,
            context_key=context_key,
            query_key=query_key,
            answer_key=answer_key,
            model=model,
            reference_key=reference_key,
            user_prompt=gen_prompt,
        )

        ppl.vqascorer = Pdf2QA.PdfQAScorer(
            input_key=context_key,
            output_key=score_key,
            query_key=query_key,
            answer_key=answer_key,
            image_key=image_key,
            model=model,
            user_prompt=score_prompt,
        )

        ppl.quality_filter = Text2qa.qa_score_filter(
            input_key=score_key,
            min_score=filter_threshold,
        )

        if to_chat:
            ppl.sft_formatter = Pdf2QA.vqa_to_chat_format(
                image_key=image_key,
                query_key=query_key,
                answer_key=answer_key
            )
    return ppl
