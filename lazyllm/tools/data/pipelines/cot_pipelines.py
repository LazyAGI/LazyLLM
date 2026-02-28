from lazyllm import pipeline
from lazyllm.tools.data import genCot, mathQA


def build_cot_pipeline(
        input_key='query',
        reference_key='reference',
        cot_key='cot_answer',
        extracted_key='llm_extracted',
        verify_key='is_equal',

        model=None,
        use_self_consistency=False,
        num_samples=5,
        user_prompt=None,

        enable_verify=True,
        enable_filter_correct=False,
):
    with pipeline() as ppl:
        if use_self_consistency:
            ppl.generator = genCot.SelfConsistencyCoTGenerator(
                input_key=input_key,
                output_key=cot_key,
                num_samples=num_samples,
                model=model,
                user_prompt=user_prompt,
            )
        else:
            ppl.generator = genCot.CoTGenerator(
                input_key=input_key,
                output_key=cot_key,
                model=model,
                user_prompt=user_prompt,
            )

        ppl.extractor = mathQA.math_answer_extractor(
            input_key=cot_key,
            output_key=extracted_key,
        )

        if enable_verify:
            ppl.verify = genCot.answer_verify(
                answer_key=reference_key,
                infer_key=extracted_key,
                output_key=verify_key,
            )
            ppl.filter_wrong_answer = genCot.wrong_filter(input_key=verify_key)

    return ppl
