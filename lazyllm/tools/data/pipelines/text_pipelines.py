from lazyllm import pipeline
from lazyllm.tools.data import Text2qa


def build_text2qa_pipeline(
        text_key='text',
        chunk_key='chunk',
        instruction_key='instruction',
        output_key='output',
        model=None,
        score_prompt=None,
        tokenizer=None,
        chunk_size=200,
        tokenize=False,
        qa_prompt=None,
        threshold=1):

    with pipeline() as ppl:

        ppl.text_to_chunks = Text2qa.TextToChunks(
            input_key=text_key,
            output_key=chunk_key,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            tokenize=tokenize
        )

        ppl.noise_filter = Text2qa.empty_or_noise_filter(
            input_key=chunk_key
        )

        ppl.invalid_unicode_cleaner = Text2qa.invalid_unicode_cleaner(
            input_key=chunk_key
        )

        ppl.generate_qa = Text2qa.ChunkToQA(
            input_key=chunk_key,
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            user_prompt=qa_prompt
        )

        ppl.qa_scorer = Text2qa.QAScorer(
            input_key=chunk_key,
            output_key='score',
            query_key=instruction_key,
            answer_key=output_key,
            model=model,
            user_prompt=score_prompt
        )

        ppl.score_filter = Text2qa.qa_score_filter(
            input_key='score',
            min_score=threshold
        )

        ppl.sft_data = Text2qa.to_alpaca_sft(
            query_key=instruction_key,
            context_key=chunk_key,
            answer_key=output_key
        )

    return ppl
