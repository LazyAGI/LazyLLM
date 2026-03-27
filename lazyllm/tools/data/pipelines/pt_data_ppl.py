from lazyllm import pipeline
from lazyllm.tools.data import pt


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
