"""Test cases for AgenticRAG operators"""
import os
import shutil
import json
from unittest.mock import Mock, MagicMock
import pytest
import lazyllm
from lazyllm import config
from lazyllm.tools.data import agenticrag

class TestAgenticRAGOperators:
    """Test suite for AgenticRAG data operators"""

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
    def model_set(self):
        # self.llm = lazyllm.TrainableModule('Qwen2.5-0.5B-Instruct')
        self.llm = lazyllm.OnlineChatModule()
        return self.llm
    def test_qaf1_sample_evaluator_basic(self):
        """Test F1 evaluator with basic inputs"""
        func = agenticrag.AgenticRAGQAF1SampleEvaluator()
        # Test data with exact matches
        inputs = [
            {
                'refined_answer': 'The capital of France is Paris',
                'golden_doc_answer': 'The capital of France is Paris'
            },
            {
                'refined_answer': 'Python is a programming language',
                'golden_doc_answer': 'Python is a programming language'
            }
        ]
        
        res = func(inputs)
        
        # Check that F1 scores are added
        assert len(res) == 2
        assert all('F1Score' in item for item in res)
        # Exact matches should have F1 score of 1.0
        assert res[0]['F1Score'] == 1.0
        assert res[1]['F1Score'] == 1.0

    def test_qaf1_sample_evaluator_partial_match(self):
        """Test F1 evaluator with partial matches"""

        func = agenticrag.AgenticRAGQAF1SampleEvaluator()
        
        inputs = [
            {
                'refined_answer': 'Paris is the capital of France',
                'golden_doc_answer': 'The capital is Paris'
            }
        ]
        
        res = func(inputs)
        
        # Partial match should have F1 score between 0 and 1
        assert 0 < res[0]['F1Score'] < 1.0

    def test_qaf1_sample_evaluator_multiple_ground_truths(self):
        """Test F1 evaluator with multiple ground truth answers"""
        func = agenticrag.AgenticRAGQAF1SampleEvaluator()
        
        inputs = [
            {
                'refined_answer': 'Paris',
                'golden_doc_answer': ['Paris', 'Paris, France', 'The city of Paris']
            }
        ]
        
        res = func(inputs)
        
        # Should match at least one ground truth perfectly
        assert res[0]['F1Score'] == 1.0

    def test_qaf1_sample_evaluator_custom_keys(self):
        """Test F1 evaluator with custom field names"""
        func = agenticrag.AgenticRAGQAF1SampleEvaluator(
            prediction_key='my_prediction',
            ground_truth_key='my_truth',
            output_key='my_score'
        )
        
        inputs = [
            {
                'my_prediction': 'Answer A',
                'my_truth': 'Answer A'
            }
        ]
        
        res = func(inputs)
        
        assert 'my_score' in res[0]
        assert res[0]['my_score'] == 1.0

    def test_qaf1_sample_evaluator_normalize_answer(self):
        """Test answer normalization (articles, punctuation, case)"""
        func = agenticrag.AgenticRAGQAF1SampleEvaluator()
        
        # Test normalization
        normalized = func.normalize_answer("The QUICK brown fox, jumped!")
        expected = "quick brown fox jumped"
        assert normalized == expected

    def test_qaf1_sample_evaluator_none_handling(self):
        """Test F1 evaluator with None values"""
        func = agenticrag.AgenticRAGQAF1SampleEvaluator()
        
        inputs = [
            {'refined_answer': None, 'golden_doc_answer': 'Some answer'},
            {'refined_answer': 'Some answer', 'golden_doc_answer': None},
            {'refined_answer': None, 'golden_doc_answer': None},
        ]
        
        res = func(inputs)
        
        # All should have F1 score of 0
        assert all(item['F1Score'] == 0.0 for item in res)


    def test_atomic_task_generator_mock(self):
        """Test Atomic Task Generator with mocked LLM"""

        func = agenticrag.AgenticRAGAtomicTaskGenerator(
            data_num=1,
            max_per_task=3,
            max_question=3,
            llm=self.model_set().share(),
            _concurrency_mode='single'  # 使用单线程模式避免序列化问题
        )

        # 使用更丰富、更具体的测试文档内容，让 LLM 能够提取有效结论
        inputs = [{
            'prompts': '''
            龙美术馆是中国知名的私立美术馆，由收藏家刘益谦、王薇夫妇创办。
            龙美术馆目前在上海有两个馆：浦东馆位于浦东新区罗山路2255弄210号，
            于2012年12月开馆；西岸馆位于徐汇区龙腾大道3398号，于2014年3月开馆。
            龙美术馆收藏了大量中国传统艺术和当代艺术作品，包括书画、雕塑、装置艺术等。
            2025年春季，龙美术馆将举办"东方美学"主题展览，展出时间为3月1日至6月30日。
            '''
        }]
        res = func(inputs)

        # 断言：由于 LLM 输出有不确定性，只检查结构和类型
        assert isinstance(res, list)
        # 可能生成 0 到多个 QA 对，取决于 LLM 输出
        # 如果生成了结果，检查必要字段存在
        for item in res:
            assert 'question' in item
            assert 'answer' in item or 'refined_answer' in item

            


    def test_depth_qa_generator_mock(self):
        """Test Depth QA Generator with mocked LLM"""
        
        func = agenticrag.AgenticRAGDepthQAGenerator(n_rounds=1, llm=self.model_set().share())
     
        inputs = [
            {
                'question': 'What is AI?',
                'answer': 'AI is artificial intelligence',
                'refined_answer': 'AI is artificial intelligence'
            }
        ]
        
        res = func(inputs)
        
        assert isinstance(res, list)

    def test_width_qa_generator_mock(self):
        """Test Width QA Generator with mocked LLM"""
        func = agenticrag.AgenticRAGWidthQAGenerator(llm=self.model_set().share())
        
        inputs = [
            {
                'question': 'What is AI?',
                'identifier': 'AI',
                'answer': 'Artificial Intelligence'
            },
            {
                'question': 'What is ML?',
                'identifier': 'ML',
                'answer': 'Machine Learning'
            }
        ]
        
        res = func(inputs)
        
        assert isinstance(res, list)


    def test_evaluator_with_list_assertions(self):
        """Test F1 evaluator maintains list structure"""

        func = agenticrag.AgenticRAGQAF1SampleEvaluator()

        inputs = [{'refined_answer': f'Answer {i}', 'golden_doc_answer': f'Answer {i}'} for i in range(10)]
        
        # Should accept list
        res = func(inputs)
        
        # Should return list of same length
        assert isinstance(res, list)
        assert len(res) == len(inputs)
        
        # All items should have the score field
        assert all('F1Score' in item for item in res)
