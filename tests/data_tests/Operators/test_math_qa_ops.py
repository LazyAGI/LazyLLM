import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data import MathQA


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

class TestMathQAOperators:

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

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
