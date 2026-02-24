import os
import time
import pytest
import random
import shutil
import json
from lazyllm import config, LOG
from lazyllm.tools.data import demo1, demo2, data_register, genCot, EnQA, MathQA, Text2qa

class MockModel:
    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self

class MockModel:
    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self

class TestDataOperators:

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

    def test_build_pre_suffix(self):
        func = demo1.build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
        inputs = [{'text': 'world'}, {'text': 'lazyLLM'}]
        res = func(inputs)
        assert res == [{'text': 'Hello, world!'}, {'text': 'Hello, lazyLLM!'}]

    def test_process_uppercase(self):
        func = demo1.process_uppercase(input_key='text', _concurrency_mode='thread')
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        inputs = [{'text': text} for text in (['hello', 'world'] * 2000)]
        res = func(inputs)
        expected = [{'text': text.upper()} for text in (['hello', 'world'] * 2000)]
        assert sorted(res, key=lambda x: x['text']) == sorted(expected, key=lambda x: x['text'])

    def test_add_suffix(self):
        func = demo2.AddSuffix(input_key='text', suffix='!!!', _max_workers=32, _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        inputs = [{'text': text} for text in (['exciting', 'amazing'] * 2000)]
        res = func(inputs)
        expected = [{'text': text + '!!!'} for text in (['exciting', 'amazing'] * 2000)]
        assert sorted(res, key=lambda x: x['text']) == sorted(expected, key=lambda x: x['text'])

    def test_rich_content(self):
        func = demo2.rich_content(input_key='text')
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        assert res == [
            {'text': 'This is a test.'},
            {'text': 'This is a test. - part 1'},
            {'text': 'This is a test. - part 2'}]

    def test_output_file(self):
        func = demo2.rich_content(input_key='text').set_output(self.root_dir)
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        assert isinstance(res, str)
        assert os.path.exists(res)
        assert res.endswith('.jsonl')

    def test_error_handling(self):
        op = demo2.error_prone_op(input_key='text', _save_data=True, _concurrency_mode='single')
        inputs = [{'text': 'ok1'}, {'text': 'fail'}, {'text': 'ok2'}]
        res = op(inputs)

        # Check results - failure should be skipped in valid results
        assert len(res) == 2
        assert res[0]['text'] == 'Processed: ok1'
        assert res[1]['text'] == 'Processed: ok2'

        # Check error file
        err_file = op._store.error_path
        assert os.path.exists(err_file)
        with open(err_file, 'r', encoding='utf-8') as f:
            errs = [json.loads(line) for line in f]
            assert len(errs) == 1
            assert errs[0]['text'] == 'fail'
            assert 'Intentional error for testing.' in errs[0]['infer_error']

    def test_process_safety_stress(self):
        # Test if multiple workers/instances cause file corruption or data loss
        count = 1000
        inputs = [{'text': f'id_{i}'} for i in range(count)]
        # Use a high number of workers to increase contention probability
        op = demo2.AddSuffix(input_key='text', suffix='_safe', _max_workers=32, _concurrency_mode='process')

        res = op(inputs)

        # 1. Check results count
        assert len(res) == count

        # 2. Check file integrity (no partial JSON writes)
        load_res = []
        with open(op._store.save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == count
            for line in lines:
                # If json.loads fails, it means process safety failed on IO
                data = json.loads(line)
                assert data['text'].endswith('_safe')
                load_res.append(data)
        # 3. Check all expected entries are present
        sorted_res = sorted(res, key=lambda x: x['text'])
        sorted_load = sorted(load_res, key=lambda x: x['text'])
        assert sorted_res == sorted_load

    @pytest.mark.skip(reason='Long running test')
    def test_dummy_llm_operator(self):
        num_qa = 60000

        @data_register('data.demo1', rewrite_func='forward', _concurrency_mode='thread')
        def dummy_llm_op(data, input_key='text', output_key='llm_output'):
            assert isinstance(data, dict)
            content = data.get(input_key, '')
            time.sleep(random.uniform(2, 12))  # Simulate LLM latency with variability
            data[output_key] = f'LLM response for: {content}'
            return data

        llm_func = demo1.dummy_llm_op(input_key='text', output_key='llm_output')
        assert llm_func._concurrency_mode == 'thread'
        inputs = [{'text': f'query_{i}', 'id': i} for i in range(num_qa)]
        res = llm_func(inputs)

        assert len(res) == num_qa

        sorted_res = sorted(res, key=lambda x: x['id'])
        for i, item in enumerate(sorted_res):
            expected_text = f'query_{i}'
            expected_llm = f'LLM response for: {expected_text}'
            assert item['text'] == expected_text
            assert item['llm_output'] == expected_llm

        load_res = []
        with open(llm_func._store.save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == num_qa
            for line in lines:
                data = json.loads(line)
                load_res.append(data)
        sorted_load = sorted(load_res, key=lambda x: x['id'])
        assert sorted_res == sorted_load

    # ===============================
    # cot_ops
    # ===============================

    def test_cot_generator(self):
        model = MockModel({'query': 'What is 2+2?', 'cot_answer': 'XXXXX \\boxed{4}'})

        op = genCot.CoTGenerator(
            input_key='query',
            output_key='cot_answer',
            model=model
        )

        data = {'query': 'What is 2+2?'}

        res = op(data)

        assert 'cot_answer' in res[0]

    def test_self_consistency(self):
        model = MockModel('XXXXX \\boxed{9}')

        op = genCot.SelfConsistencyCoTGenerator(
            input_key='query',
            output_key='cot_answer',
            num_samples=2,
            model=model
        )

        data = {'query': 'What is 3*3?'}

        res = op(data)

        assert 'cot_answer' in res[0]

    def test_answer_verify(self):
        op = genCot.answer_verify(
            answer_key='reference',
            infer_key='llm_extracted',
            output_key='is_equal'
        )

        data = {
            'reference': '1/2',
            'llm_extracted': '0.5'
        }

        res = op(data)

        assert 'is_equal' in res[0]

    def test_query_rewriter(self):
        model = MockModel({'query': 'What is AI?', 'rewrite_querys': ['Ai is xxx', 'Ai is yyyy']})
        rewrite_num = 2
        op = EnQA.QueryRewriter(
            input_key='query',
            output_key='rewrite_querys',
            rewrite_num=rewrite_num,
            model=model
        )

        data = {'query': 'What is AI?'}

        res = op(data)

        assert 'rewrite_querys' in res[0]
        assert len(res[0]['rewrite_querys']) == 2

    def test_diversity_scorer(self):
        model = MockModel({'rewrite_querys': ['a', 'b'], 'diversity_querys': [1, 0]})

        op = EnQA.DiversityScorer(
            input_key='rewrite_querys',
            output_key='diversity_querys',
            model=model
        )

        data = {'rewrite_querys': ['a', 'b']}

        res = op(data)

        assert 'diversity_querys' in res[0]
        assert len(res[0]['diversity_querys']) == 2

    def test_post_processor(self):
        op = EnQA.post_processor(
            input_key='diversity_querys'
        )

        data = {
            'diversity_querys': [
                {
                    'rewritten_query': 'a',
                    'diversity_score': 1
                }
            ]
        }

        res = op(data)

        assert isinstance(res, list)

    def test_diversity_filter(self):
        op = EnQA.diversity_filter(
            input_key='score',
            min_score=1
        )

        data = {'score': 0}

        res = op(data)

        assert res == []

    def test_math_answer_extractor(self):
        op = MathQA.math_answer_extractor(
            input_key='answer',
            output_key='math_answer'
        )

        data = {'answer': 'result is \\boxed{42}'}

        res = op(data)

        assert 'math_answer' in res[0]
        assert res[0]['math_answer'] == '42'

    def test_difficulty_evaluator(self):
        model = MockModel({'question': '1+1', 'difficulty': 'Easy'})

        op = MathQA.DifficultyEvaluator(
            input_key='question',
            output_key='difficulty',
            model=model
        )

        data = {'question': '1+1'}

        res = op(data)

        assert 'difficulty' in res[0]
        assert res[0]['difficulty'] == 'Easy'

    def test_difficulty_batch(self):
        op = MathQA.DifficultyEvaluatorBatch(
            input_key='difficulty'
        )

        data = [
            {'difficulty': 'Easy'},
            {'difficulty': 'Hard'}
        ]

        res = op(data)

        assert isinstance(res, list)

    def test_quality_evaluator(self):
        model = MockModel({
            'question': 'Q',
            'answer': 'A',
            'score': 0
        })

        op = MathQA.QualityEvaluator(
            question_key='question',
            answer_key='answer',
            output_key='score',
            model=model
        )

        data = {
            'question': '1 + 1 = ?',
            'answer': '3'
        }

        res = op(data)

        assert 'score' in res[0]
        assert res[0]['score'] == 0

    def test_duplicate_answer_detector(self):
        op = MathQA.DuplicateAnswerDetector(
            question_key='question',
            answer_key='answer',
            output_key='duplicate'
        )

        data = {
            'question': 'Q',
            'answer': 'A' * 50
        }

        res = op(data)

        assert 'duplicate' in res[0]

    def test_token_length_filter(self):
        op = MathQA.ReasoningAnswerTokenLengthFilter(
            input_key='answer',
            max_answer_token_length=10,
            tokenize=True
        )

        data = [{'answer': 'short'}]

        res = op(data)

        assert len(res) == 1

    def test_question_fusion(self):
        excepted_output = {
            'question_list': [
                {'question': '1+1', 'answer': '2'},
                {'question': '2+2', 'answer': '4'}
            ],
            'question': '1 + 1 + 2 + 2 = ?',
            'answer': '6'
        }
        model = MockModel(excepted_output)

        op = MathQA.QuestionFusionGenerator(
            list_key='question_list',
            input_key='question',
            output_key='answer',
            model=model
        )

        data = {
            'question_list': [
                {'question': '1+1', 'answer': '2'},
                {'question': '2+2', 'answer': '4'}
            ]
        }

        res = op(data)

        assert 'answer' in res[0]
        assert 'question' in res[0]
    # text2qa_ops tests
    def test_text2qa_text_to_chunks(self):
        # Example from data.operators.text2qa_ops.TextToChunks
        op = Text2qa.TextToChunks(input_key='content', output_key='chunk', chunk_size=10, tokenize=False)
        data = [{'content': 'line1\nline2\nline3\nline4'}]
        res = op(data)
        assert res == [
            {'content': 'line1\nline2\nline3\nline4', 'chunk': 'line1\nline2'},
            {'content': 'line1\nline2\nline3\nline4', 'chunk': 'line3\nline4'},
        ]

    def test_text2qa_empty_or_noise_filter(self):
        # Example from data.operators.text2qa_ops.empty_or_noise_filter
        op = Text2qa.empty_or_noise_filter(input_key='chunk')
        data = [{'chunk': 'hello'}, {'chunk': ''}, {'chunk': '\n'}]
        res = op(data)
        assert res == [{'chunk': 'hello'}]

    def test_text2qa_invalid_unicode_cleaner(self):
        # Example from data.operators.text2qa_ops.invalid_unicode_cleaner
        op = Text2qa.invalid_unicode_cleaner(input_key='chunk')
        data = [{'chunk': 'valid text\uFFFE tail'}]
        res = op(data)
        assert res == [{'chunk': 'valid text tail'}]

    def test_text2qa_chunk_to_qa(self):
        llm = MockModel({'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！'})
        op = Text2qa.ChunkToQA(input_key='chunk', query_key='query', answer_key='answer', model=llm)
        data = [{'chunk': '今天是晴天！'}]
        res = op(data)
        assert len(res) == 1
        assert 'chunk' in res[0] and 'query' in res[0] and 'answer' in res[0]

    def test_text2qa_qa_scorer(self):
        # Example from data.operators.text2qa_ops.QAScorer
        llm = MockModel({'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！', 'score': 1})
        op = Text2qa.QAScorer(input_key='chunk', output_key='score', query_key='query', answer_key='answer', model=llm)
        data = [
            {'chunk': '今天是晴天！', 'query': '今天的天气怎么样？', 'answer': '今天是晴天！'}
        ]
        res = op(data)
        assert len(res) == 1
        assert res[0].get('score') == 1
