from lazyllm import pipeline
from lazyllm.tools.data import Pdf2Qa, Text2qa

def build_pdf2qa_pipeline(
    model,
    mineru_api,
    pdf_input_key='pdf_path',
    chunk_key='chunk',
    instruction_key='instruction',
    output_key='output',
    score_key='score',
    image_key='image_path',
    reader_upload_mode=True,
    reader_use_cache=True,
    qa_user_prompt=None,
    scorer_user_prompt=None,
    threshold=1,
    image_output_folder='./images',
    image_size=336,  # accept both int and tuple
    max_chunk_chars=1500,
    chat_format=True,
    context_key='context'
):

    with pipeline() as ppl:
        ppl.PdfProcessor = Pdf2Qa.PdfProcessor(
            input_key=pdf_input_key,
            output_key=chunk_key,
            reader_url=mineru_api,
            upload_mode=reader_upload_mode,
            use_cache=reader_use_cache,
            image_output_folder=image_output_folder,
            image_key=image_key,
            image_size=image_size,
            max_chunk_chars=max_chunk_chars,
            context_key=context_key
        )

        ppl.qa_generator = Pdf2Qa.ImageToVQA(
            image_key=image_key,
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            user_prompt=qa_user_prompt,
            context_key=chunk_key
        )

        ppl.qa_scorer = Pdf2Qa.PdfQAScorer(
            input_key=chunk_key,
            output_key=score_key,
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            user_prompt=scorer_user_prompt,
            image_key=image_key
        )

        ppl.filter = Text2qa.qa_score_filter(
            input_key=score_key,
            min_score=threshold
        )

        if chat_format:
            ppl.vqa_formatter = Pdf2Qa.vqa_to_chat_format(
                image_key=image_key,
                query_key=instruction_key,
                answer_key=output_key
            )
        else:
            ppl.alpaca_formatter = Text2qa.to_alpaca_sft(
                query_key=instruction_key,
                answer_key=output_key,
                context_key=context_key
            )

    return ppl
