"""Tests for RAG Pipeline.

参考 test_data_pipeline.py 的写法，传入 data 并验证 res 输出。
- qa_evaluation_pipeline: 不需要 LLM，使用默认 key 与 pipeline 一致。
- 需要 LLM 的 pipeline：传入 llm=None 时仅校验可构建，执行 ppl(data) 时期望 ValueError。
"""

import os
import shutil
import pytest
from lazyllm import config
from lazyllm.tools.data.pipelines.rag_pipeline import (
    atomic_rag_pipeline,
    depth_qa_single_round_pipeline,
    depth_qa_pipeline,
    qa_evaluation_pipeline,
    width_qa_pipeline,
)


class TestRagPipeline:
    """Tests for all RAG pipelines with actual data execution."""

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
        """qa_evaluation_pipeline 默认使用 prediction_key='re_answer', ground_truth_key='golden_answer', output_key='F1Score'。"""
        ppl = qa_evaluation_pipeline()
        # data 的 key 必须与 pipeline 默认一致
        data = [
            {'re_answer': 'The answer is 42', 'golden_answer': 'The answer is 42'},
            {'re_answer': 'Python is great', 'golden_answer': 'Python is awesome'},
        ]
        res = ppl(data)
        assert isinstance(res, list)
        assert len(res) == 2
        # pipeline 默认 output_key 为 'F1Score'，算子参数为 result_key
        assert 'F1Score' in res[0]
        assert 'F1Score' in res[1]

    def test_qa_evaluation_pipeline_custom_keys(self):
        """使用自定义 key 时，data 与 pipeline 参数一致。"""
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
        """atomic_rag_pipeline 需要 LLM；llm=None 时前段算子会跑完，全量算子 GroupAndLimit 会收到空列表。"""
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
        # llm=None 时前面单条算子可能因 LLM 未配置而丢弃或报错，最终多为空或少量结果
        assert isinstance(res, list)

    def test_depth_qa_single_round_pipeline(self):
        """depth_qa_single_round_pipeline 需要 LLM；llm=None 时执行会得到空列表（中间算子丢弃）。"""
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
        """depth_qa_pipeline 返回可调用函数，需要 LLM；llm=None 时执行会得到空列表。"""
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
        """width_qa_pipeline 返回可调用函数，需要 LLM；llm=None 时执行会在 MergePairs 报错。"""
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
        """width_qa 至少需要 2 条数据；1 条时在 merge 前直接返回空列表，不调用 LLM。"""
        ppl_fn = width_qa_pipeline(llm=None)
        assert callable(ppl_fn)
        single_item_data = [
            {'question': 'What is ML?', 'identifier': 'doc1', 'answer': 'Machine Learning'}
        ]
        res = ppl_fn(single_item_data)
        assert isinstance(res, list)
        assert len(res) == 0
