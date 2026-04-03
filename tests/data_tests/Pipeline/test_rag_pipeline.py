import os
import shutil
import pytest
from lazyllm import config
from lazyllm.tools.data.pipelines.rag_pipelines import (
    atomic_rag_pipeline,
    depth_qa_single_round_pipeline,
    depth_qa_pipeline,
    qa_evaluation_pipeline,
    width_qa_pipeline,
)


class TestRagPipeline:

    def setup_method(self):
        self.root_dir = './test_rag_pipeline'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_qa_evaluation_pipeline(self):
        ppl = qa_evaluation_pipeline()
        data = [
            {'re_answer': 'The answer is 42', 'golden_answer': 'The answer is 42'},
            {'re_answer': 'Python is great', 'golden_answer': 'Python is awesome'},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 2
        assert 'F1Score' in res[0]
        assert 'F1Score' in res[1]

    def test_qa_evaluation_pipeline_custom_keys(self):
        ppl = qa_evaluation_pipeline(
            prediction_key='pred',
            ground_truth_key='gold',
            output_key='f1',
        )
        data = [
            {'pred': 'yes', 'gold': 'yes'},
            {'pred': 'no', 'gold': 'no'},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 2
        assert 'f1' in res[0]
        assert 'f1' in res[1]

    def test_atomic_rag_pipeline(self):
        ppl = atomic_rag_pipeline(
            llm=None,
            input_key='content',
            max_per_task=5,
            max_question=10,
        )
        data = [
            {'content': 'This is the first document'},
            {'content': 'This is the second document'},
        ]
        res = ppl(data)
        assert isinstance(res, list)

    def test_depth_qa_single_round_pipeline(self):
        ppl = depth_qa_single_round_pipeline(
            llm=None,
            identifier_key='my_identifier',
            new_identifier_key='my_new_id',
            relation_key='my_relation',
            question_key='my_question',
        )
        data = [
            {'my_identifier': 'What is ML?', 'text': 'ML content 1'},
            {'my_identifier': 'What is DL?', 'text': 'DL content 2'},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) <= len(data)

    def test_depth_qa_pipeline(self):
        ppl_fn = depth_qa_pipeline(
            llm=None,
            input_key='document',
            output_key='depth_q',
            n_rounds=2,
        )
        assert callable(ppl_fn)
        data = [
            {'document': 'Document 1 about ML'},
            {'document': 'Document 2 about DL'},
        ]
        res = ppl_fn(data)
        assert isinstance(res, list)
        assert len(res) <= len(data)

    def test_width_qa_pipeline(self):
        ppl_fn = width_qa_pipeline(
            llm=None,
            input_question_key='my_q',
            input_identifier_key='my_doc_id',
            input_answer_key='my_ans',
            output_question_key='my_output_q',
        )
        assert callable(ppl_fn)
        data = [
            {'my_q': 'What is ML?', 'my_doc_id': 'doc1', 'my_ans': 'Machine Learning'},
            {'my_q': 'What is DL?', 'my_doc_id': 'doc2', 'my_ans': 'Deep Learning'},
        ]
        with pytest.raises(ValueError, match='LLM is not configured'):
            ppl_fn(data)

    def test_width_qa_pipeline_requires_two_items(self):
        ppl_fn = width_qa_pipeline(llm=None)
        assert callable(ppl_fn)
        single_item_data = [
            {'question': 'What is ML?', 'identifier': 'doc1', 'answer': 'Machine Learning'}
        ]
        res = ppl_fn(single_item_data)
        assert isinstance(res, list)
        assert len(res) == 0
