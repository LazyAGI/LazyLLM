from lazyllm import pipeline
from lazyllm.tools.data import genCot, mathQA, Text2qa


def build_math_cot_pipeline(
    question_key='question',
    reference_key='reference',
    answer_key='answer',
    extracted_key='math_answer',
    verify_key='is_equal',

    model=None,
    num_samples=3,
    cot_user_prompt=None,

    max_answer_token_length=10000,
    tokenize=False,
    tokenizer=None,

    min_repeat_len=25,
    repeat_threshold=3,
    periodic_min_repeat=3,

    quality_user_prompt=None,
    difficulty_user_prompt=None,
    qa_scorer=False,
    difficluty_evaluator=False
):
    with pipeline() as ppl:
        ppl.cot_generator = genCot.SelfConsistencyCoTGenerator(
            input_key=question_key,
            output_key=answer_key,
            num_samples=num_samples,
            model=model,
            user_prompt=cot_user_prompt,
        )

        ppl.extractor = mathQA.boxed_answer_extractor(
            input_key=answer_key,
            output_key=extracted_key,
        )

        ppl.verify = genCot.answer_verify(
            answer_key=reference_key,
            infer_key=extracted_key,
            output_key=verify_key,
        )

        ppl.length_filter = mathQA.ReasoningAnswerTokenLengthFilter(
            input_key=answer_key,
            max_answer_token_length=max_answer_token_length,
            tokenize=tokenize,
            tokenizer=tokenizer,
        )

        ppl.dup_detector = mathQA.DuplicateAnswerDetector(
            question_key=question_key,
            answer_key=answer_key,
            min_repeat_len=min_repeat_len,
            repeat_threshold=repeat_threshold,
            periodic_min_repeat=periodic_min_repeat,
        )

        if qa_scorer:
            ppl.quality = mathQA.QualityEvaluator(
                question_key=question_key,
                answer_key=answer_key,
                user_prompt=quality_user_prompt,
                output_key='quality_score',
                model=model
            )

            ppl.quality_filter = Text2qa.qa_score_filter(
                input_key='quality_score',
                min_score=1
            )

        if difficluty_evaluator:
            ppl.difficulty = mathQA.DifficultyEvaluator(
                input_key=question_key,
                user_prompt=difficulty_user_prompt,
                model=model
            )

        ppl.wrong_fiilter = genCot.wrong_filter(input_key=verify_key)
        ppl.to_sft = Text2qa.to_alpaca_sft(
            query_key=question_key,
            context_key=reference_key,
            answer_key=answer_key
        )

    return ppl
