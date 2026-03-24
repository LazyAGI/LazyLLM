from lazyllm import pipeline
from lazyllm.tools.data import pt


def build_long_context_pipeline(llm, context_key='context', question_key='question',
                                   answer_key='answer', num_distractors=3, seed=None):
    with pipeline() as ppl:
        ppl.context_expansion = pt.ContextExpansion(
            llm=llm,
            context_key=context_key,
            question_key=question_key,
            answer_key=answer_key,
        )
        ppl.context_reconstruction = pt.ContextReconstruction(
            context_key='expanded_context',
            question_key=question_key,
            answer_key=answer_key,
            num_distractors=num_distractors,
            seed=seed,
        )
    return ppl
