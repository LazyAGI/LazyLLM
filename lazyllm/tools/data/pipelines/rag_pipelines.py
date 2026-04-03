from typing import List, Optional
from lazyllm import pipeline
from lazyllm import LOG
from lazyllm.tools.data import agenticrag

def atomic_rag_pipeline(
    llm,
    input_key='text',
    max_per_task=10,
    max_question=10,
    llm_verify_filter_threshold=1,
    max_optional_variants=20,
):
    with pipeline() as ppl:
        ppl.get_identifier = agenticrag.AgenticRAGGetIdentifier(llm=llm, input_key=input_key)
        ppl.get_conclusion = agenticrag.AgenticRAGGetConclusion(llm=llm, input_key=input_key)
        ppl.expand = agenticrag.AgenticRAGExpandConclusions(max_per_task=max_per_task)
        ppl.generate_question = agenticrag.AgenticRAGGenerateQuestion(llm=llm)
        ppl.clean_qa = agenticrag.AgenticRAGCleanQA(llm=llm)
        ppl.llm_verify = agenticrag.AgenticRAGLLMVerify(llm=llm, filter_threshold=llm_verify_filter_threshold)
        ppl.golden_doc = agenticrag.AgenticRAGGoldenDocAnswer(llm=llm, input_key=input_key)
        ppl.optional = agenticrag.AgenticRAGOptionalAnswers(llm=llm, max_variants=max_optional_variants)
        ppl.group = agenticrag.AgenticRAGGroupAndLimit(input_key=input_key, max_question=max_question)

    return ppl


def depth_qa_single_round_pipeline(
    llm,
    identifier_key='identifier',
    new_identifier_key='new_identifier',
    relation_key='relation',
    question_key='depth_question',
    depth_verify_filter_threshold=1,
):
    with pipeline() as ppl:
        ppl.backward = agenticrag.DepthQAGBackwardTask(
            llm=llm,
            identifier_key=identifier_key,
            new_identifier_key=new_identifier_key,
            relation_key=relation_key
        )
        ppl.check = agenticrag.DepthQAGCheckSuperset(
            llm=llm,
            new_identifier_key=new_identifier_key,
            relation_key=relation_key,
            identifier_key=identifier_key
        )
        ppl.generate = agenticrag.DepthQAGGenerateQuestion(
            llm=llm,
            new_identifier_key=new_identifier_key,
            relation_key=relation_key,
            identifier_key=identifier_key,
            question_key=question_key
        )
        ppl.verify = agenticrag.DepthQAGVerifyQuestion(
            llm=llm,
            question_key=question_key,
            filter_threshold=depth_verify_filter_threshold,
        )

    return ppl


def depth_qa_pipeline(
    llm,
    input_key='text',
    output_key='question',
    n_rounds=1,
    depth_verify_filter_threshold=1,
):
    def pipeline_fn(data_list):
        # Step 1: Get initial identifiers
        with pipeline() as ppl_identifier:
            ppl_identifier.get_id = agenticrag.DepthQAGGetIdentifier(llm=llm, input_key=input_key)
        data_list = ppl_identifier(data_list)

        if not data_list:
            LOG.warning('No data after getting identifiers.')
            return []

        # Iterative depth question generation for each round
        for round_id in range(1, n_rounds + 1):
            identifier_key = 'identifier' if round_id == 1 else f'new_identifier_{round_id - 1}'
            new_identifier_key = f'new_identifier_{round_id}'
            relation_key = f'relation_{round_id}'
            question_key = f'{output_key}_{round_id}'

            LOG.info(f'Starting round {round_id}: identifier_key={identifier_key}, question_key={question_key}')

            # Build and execute pipeline for this round
            with pipeline() as ppl_round:
                ppl_round.backward = agenticrag.DepthQAGBackwardTask(
                    llm=llm,
                    identifier_key=identifier_key,
                    new_identifier_key=new_identifier_key,
                    relation_key=relation_key
                )
                ppl_round.check = agenticrag.DepthQAGCheckSuperset(
                    llm=llm,
                    new_identifier_key=new_identifier_key,
                    relation_key=relation_key,
                    identifier_key=identifier_key
                )
                ppl_round.generate = agenticrag.DepthQAGGenerateQuestion(
                    llm=llm,
                    new_identifier_key=new_identifier_key,
                    relation_key=relation_key,
                    identifier_key=identifier_key,
                    question_key=question_key
                )
                ppl_round.verify = agenticrag.DepthQAGVerifyQuestion(
                    llm=llm,
                    question_key=question_key,
                    filter_threshold=depth_verify_filter_threshold,
                )

            data_list = ppl_round(data_list)

            if not data_list:
                LOG.warning(f'No data left after round {round_id}.')
                break

            LOG.info(f'Round {round_id} completed. Remaining items: {len(data_list)}')

        LOG.info(f'Depth QA completed! Final count: {len(data_list)}')
        return data_list

    return pipeline_fn


def qa_evaluation_pipeline(prediction_key='re_answer', ground_truth_key='golden_answer', output_key='F1Score'):
    with pipeline() as ppl:
        ppl.normalize = agenticrag.qaf1_normalize_texts(
            predicted_key=prediction_key,
            reference_key=ground_truth_key
        )
        ppl.calculate = agenticrag.qaf1_calculate_score(result_key=output_key)

    return ppl


def width_qa_pipeline(
    llm=None,
    input_question_key: str = 'question',
    input_identifier_key: str = 'identifier',
    input_answer_key: str = 'answer',
    output_question_key: str = 'generated_width_task',
    check_require_state_one: bool = False,
    width_filter_threshold: Optional[int] = None,
    merge_pair_stride: int = 1,
    merge_max_pairs: Optional[int] = None,
    merge_max_workers: int = 8,
):
    def pipeline_fn(data: List[dict]):
        # Step 1: Prepare input batch with key mapping
        LOG.info('Preparing input batch...')
        input_batch = []
        for i, item in enumerate(data):
            input_batch.append({
                'index': i,
                'question': item.get(input_question_key, ''),
                'content_identifier': item.get(input_identifier_key, ''),
                'golden_answer': item.get(input_answer_key, '')
            })

        if len(input_batch) < 2:
            LOG.warning('Need at least 2 items to merge. Returning empty list.')
            return []

        # Step 2: Execute full-batch operator outside pipeline
        LOG.info('Merging adjacent QA pairs (full-batch operation)...')
        merge_op = agenticrag.WidthQAGMergePairs(
            llm=llm,
            pair_stride=merge_pair_stride,
            max_merge_pairs=merge_max_pairs,
            merge_max_workers=merge_max_workers,
        )
        merged_data_list = merge_op(input_batch)

        if not merged_data_list:
            LOG.warning('No valid merged questions generated.')
            return []

        LOG.info(f'{len(merged_data_list)} questions passed merge.')

        # Step 3: Process merged questions with single-item operators in pipeline
        LOG.info('Processing merged questions with pipeline (single-item operators)...')
        with pipeline() as ppl_process:
            ppl_process.check = agenticrag.WidthQAGCheckDecomposition(
                llm=llm,
                output_question_key=output_question_key,
                require_state_one=check_require_state_one,
            )
            ppl_process.verify = agenticrag.WidthQAGVerifyQuestion(
                llm=llm,
                output_question_key=output_question_key
            )
            ppl_process.filter = agenticrag.WidthQAGFilterByScore(
                llm=llm,
                filter_threshold=width_filter_threshold,
            )

        result_list = ppl_process(merged_data_list)

        LOG.info(f'Width QA generation completed! Final count: {len(result_list)}')
        return result_list

    return pipeline_fn
