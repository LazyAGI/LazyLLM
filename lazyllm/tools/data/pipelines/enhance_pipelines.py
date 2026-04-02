from lazyllm import pipeline, OnlineChatModule
from lazyllm.tools.data import enQA, Text2qa

def build_enhance_qa_pipeline(
        query_key='instruction',
        answer_key='output',
        source_key='context',
        rewrite_key='rewrite_querys',
        diversity_key='diversity_querys',
        model=None,
        rewrite_prompt=None,
        diversity_scorer_prompt=None,
        rewrite_num=3,
        diversity_score=0.5,
        qa_scorer=False):

    if model is None:
        model = OnlineChatModule()

    with pipeline() as ppl:
        ppl.query_rewriter = enQA.QueryRewriter(
            input_key=query_key,
            output_key=rewrite_key,
            rewrite_num=rewrite_num,
            model=model,
            user_prompt=rewrite_prompt
        )

        ppl.diversity_scorer = enQA.DiversityScorer(
            input_key=rewrite_key,
            output_key=diversity_key,
            model=model,
            user_prompt=diversity_scorer_prompt

        )

        ppl.post_process = enQA.post_processor(
            input_key=diversity_key
        )

        ppl.diversity_filter = enQA.diversity_filter(
            input_key='diversity_score',
            min_score=diversity_score
        )

        if qa_scorer:
            ppl.qa_scorer = Text2qa.QAScorer(
                input_key=source_key,
                query_key='rewritten_query',
                answer_key=answer_key,
                output_key='qa_score',
                model=model
            )

            ppl.qa_score_filter = Text2qa.qa_score_filter(
                input_key='qa_score',
                min_score=1
            )

        ppl.sft_data = Text2qa.to_alpaca_sft(
            query_key='rewritten_query',
            context_key=source_key,
            answer_key=answer_key
        )

    return ppl
