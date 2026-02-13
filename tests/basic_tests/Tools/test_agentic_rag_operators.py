import json
import os
import shutil
from lazyllm import config
from lazyllm.tools.data import agenticrag

class MockLLMServe:

    def __init__(self, return_value=None, raise_exc=False):
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.started = False

    def start(self):
        self.started = True
        return self

    def prompt(self, system_prompt):
        return self

    def formatter(self, formatter):
        return self

    def __call__(self, prompt):
        if self._raise_exc:
            raise RuntimeError('mock error')
        return self._return_value


class MockLLM:

    def __init__(self, answer_return=None, score_return=None,
                 answer_raise=False, score_raise=False):
        self.answer_serve = MockLLMServe(answer_return, answer_raise)
        self.score_serve = MockLLMServe(score_return, score_raise)
        self._share_count = 0

    def share(self, prompt=None, format=None, stream=None, history=None):
        # First share returns answer_serve, second returns score_serve
        if format is not None and self._share_count == 1:
            self.score_serve.formatter(format)
        if self._share_count == 0:
            self._share_count += 1
            return self.answer_serve
        else:
            return self.score_serve


class TestAgenticRAGOperators:
    def setup_method(self):
        self.root_dir = './test_data_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
    # ========== Atomic Task Generator Operators ==========

    def test_agenticrag_get_identifier(self):
        mock_llm = MockLLM(answer_return={'content_identifier': 'test_id'})
        op = agenticrag.AgenticRAGGetIdentifier(llm=mock_llm, input_key='prompts')
        result = op([{'prompts': 'test content'}])[0]
        assert result['identifier'] == 'test_id'

    def test_agenticrag_get_conclusion(self):
        mock_llm = MockLLM(answer_return=json.dumps([{'conclusion': 'C1', 'R': 'R1'}]))
        op = agenticrag.AgenticRAGGetConclusion(llm=mock_llm, input_key='prompts')
        result = op([{'prompts': 'test'}])[0]
        assert 'raw_conclusion' in result

    def test_agenticrag_expand_conclusions(self):
        op = agenticrag.AgenticRAGExpandConclusions(max_per_task=2)
        data = {
            'raw_conclusion': json.dumps([{'conclusion': 'C1', 'R': 'R1'}, {'conclusion': 'C2', 'R': 'R2'}]),
            'identifier': 'test_id'
        }
        result = op([data])
        assert len(result) == 2

    def test_agenticrag_generate_question(self):
        mock_llm = MockLLM(answer_return={'Q': 'What is test?'})
        op = agenticrag.AgenticRAGGenerateQuestion(llm=mock_llm)
        data = {
            'candidate_tasks_str': json.dumps({'conclusion': 'Test', 'R': 'relation'}),
            'identifier': 'test_id'
        }
        result = op([data])[0]
        assert result['question'] == 'What is test?'

    def test_agenticrag_clean_qa(self):
        mock_llm = MockLLM(answer_return={'refined_answer': 'Cleaned answer'})
        op = agenticrag.AgenticRAGCleanQA(llm=mock_llm)
        result = op([{'question': 'Q?', 'answer': 'A'}])[0]
        assert result['refined_answer'] == 'Cleaned answer'

    def test_agenticrag_llm_verify(self):
        mock_llm = MockLLM(
            answer_return='LLM answer',
            score_return={'answer_score': 0}
        )
        op = agenticrag.AgenticRAGLLMVerify(llm=mock_llm)
        result = op([{'question': 'Q?', 'refined_answer': 'A'}])[0]
        assert 'llm_answer' in result

    def test_agenticrag_golden_doc_answer(self):
        mock_llm = MockLLM(
            answer_return='Golden answer',
            score_return={'answer_score': 1}
        )
        op = agenticrag.AgenticRAGGoldenDocAnswer(llm=mock_llm, input_key='prompts')
        result = op([{'prompts': 'Doc', 'question': 'Q?', 'refined_answer': 'A'}])[0]
        assert result['golden_doc_answer'] == 'Golden answer'

    def test_agenticrag_optional_answers(self):
        mock_llm = MockLLM(answer_return=['A1', 'A2', 'A3'])
        op = agenticrag.AgenticRAGOptionalAnswers(llm=mock_llm)
        result = op([{'refined_answer': 'Original'}])[0]
        assert result['optional_answer'] == ['A1', 'A2', 'A3']

    def test_agenticrag_group_and_limit(self):
        op = agenticrag.AgenticRAGGroupAndLimit(input_key='prompts', max_question=2)
        data = [
            {'prompts': 'Group A', 'question': 'Q1'},
            {'prompts': 'Group A', 'question': 'Q2'},
            {'prompts': 'Group A', 'question': 'Q3'},
        ]
        result = op.forward_batch_input(data)
        assert len(result) == 2

    # ========== Depth QA Generator Operators ==========

    def test_depth_qag_get_identifier(self):
        mock_llm = MockLLM(answer_return='depth_id')
        op = agenticrag.DepthQAGGetIdentifier(llm=mock_llm, input_key='question')
        result = op([{'question': 'What?'}])[0]
        assert result['identifier'] == 'depth_id'

    def test_depth_qag_backward_task(self):
        mock_llm = MockLLM(answer_return={'identifier': 'back_id', 'relation': 'back_rel'})
        op = agenticrag.DepthQAGBackwardTask(llm=mock_llm)
        result = op([{'identifier': 'orig_id'}])[0]
        assert result['new_identifier'] == 'back_id'

    def test_depth_qag_check_superset(self):
        mock_llm = MockLLM(answer_return={'new_query': 'valid'})
        op = agenticrag.DepthQAGCheckSuperset(llm=mock_llm)
        result = op([{'new_identifier': 'new', 'relation': 'rel', 'identifier': 'orig'}])[0]
        assert result['new_identifier'] == 'new'

    def test_depth_qag_generate_question(self):
        mock_llm = MockLLM(answer_return={'new_query': 'Generated question?'})
        op = agenticrag.DepthQAGGenerateQuestion(llm=mock_llm, question_key='depth_question')
        result = op([{'new_identifier': 'new', 'relation': 'rel', 'identifier': 'orig'}])[0]
        assert result['depth_question'] == 'Generated question?'

    def test_depth_qag_verify_question(self):
        mock_llm = MockLLM(
            answer_return='LLM answer',
            score_return={'answer_score': 0}
        )
        op = agenticrag.DepthQAGVerifyQuestion(llm=mock_llm, question_key='depth_question')
        result = op([{'depth_question': 'Q?', 'refined_answer': 'A'}])[0]
        assert 'llm_answer' not in result  # Cleaned up

    # ========== Width QA Generator Operators ==========

    def test_width_qag_merge_pairs(self):
        mock_llm = MockLLM(answer_return={'question': 'Merged?', 'index': [0, 1]})
        op = agenticrag.WidthQAGMergePairs(llm=mock_llm)
        data = [{'question': 'Q1?', 'golden_answer': 'A1'}, {'question': 'Q2?', 'golden_answer': 'A2'}]
        result = op.forward_batch_input(data)
        assert len(result) == 1
        assert result[0]['question'] == 'Merged?'

    def test_width_qag_check_decomposition(self):
        mock_llm = MockLLM(answer_return={'state': 1, 'complex_question': 'Decomposed?'})
        op = agenticrag.WidthQAGCheckDecomposition(llm=mock_llm)
        result = op([{'question': 'Original?', 'original_question': ['Q1?', 'Q2?']}])[0]
        assert result['state'] == 1

    def test_width_qag_verify_question(self):
        mock_llm = MockLLM(answer_return={'llm_answer': 'Verified'})
        op = agenticrag.WidthQAGVerifyQuestion(llm=mock_llm)
        result = op([{'generated_width_task': 'Test?', 'index': 0}])[0]
        assert result['llm_answer'] == 'Verified'

    def test_width_qag_filter_by_score(self):
        mock_llm = MockLLM(answer_return={'answer_score': 0})
        op = agenticrag.WidthQAGFilterByScore(llm=mock_llm)
        result = op([{'original_answer': ['A1'], 'llm_answer': 'LLM', 'state': 1}])[0]
        assert 'llm_answer' not in result

    # ========== QA F1 Evaluator Functions ==========

    def test_qaf1_normalize_texts(self):
        func = agenticrag.qaf1_normalize_texts()
        data = {'refined_answer': 'Test!', 'golden_doc_answer': 'test'}
        result = func(data)[0]
        assert result['_normalized_prediction'] == 'test'

    def test_qaf1_calculate_score(self):
        data = {'_normalized_prediction': 'test', '_normalized_ground_truths': ['test']}
        func = agenticrag.qaf1_calculate_score()
        result = func([data])[0]
        assert result['F1Score'] == 1.0
