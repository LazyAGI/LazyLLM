from lazyllm import pipeline
from lazyllm.tools.data import Pdf2Qa, Text2qa

def build_pdf2qa_pipeline(
    model,
    mineru_api,
    pdf_input_key='pdf_path',
    text_key='text',
    chunk_key='chunk',
    instruction_key='instruction',
    output_key='output',
    score_key='score',
    image_key='image_path',
    reader_upload_mode=True,
    reader_use_cache=False,
    qa_user_prompt=None,
    scorer_user_prompt=None,
    tokenizer=None,
    chunk_size=100,
    tokenize=False,
    threshold=1
):

    with pipeline() as ppl:
        ppl.pdf2md = Pdf2Qa.Pdf2Md(
            input_key=pdf_input_key,
            output_key=text_key,
            reader_url=mineru_api,
            upload_mode=reader_upload_mode,
            use_cache=reader_use_cache
        )

        ppl.text_to_chunks = Text2qa.TextToChunks(
            input_key=text_key,
            output_key=chunk_key,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            tokenize=tokenize
        )

        ppl.generate_qa = Pdf2Qa.PdfChunkToQA(
            input_key=chunk_key,
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            user_prompt=qa_user_prompt,
            mineru_api=mineru_api,
            image_key=image_key
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

    return ppl
