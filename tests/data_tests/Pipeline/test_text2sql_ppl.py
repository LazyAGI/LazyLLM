import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.text2sql_pipelines import (
    text2sql_synthetic_ppl,
    text2sql_enhanced_ppl,
)


class MockDatabaseManager:
    def __init__(self):
        self.db_type = 'sqlite'

    def list_databases(self):
        return ['test_db']

    def database_exists(self, db_id):
        return db_id == 'test_db'

    def get_create_statements_and_insert_statements(self, db_id):
        return ['CREATE TABLE users (id INT, name TEXT)'], ['INSERT INTO users VALUES (1, "alice")']

    def batch_explain_queries(self, queries):
        class Result:
            def __init__(self, success):
                self.success = success
        return [Result(True) for _ in queries]

    def batch_execute_queries(self, queries):
        class Result:
            def __init__(self, success, data=None, columns=None):
                self.success = success
                self.data = data or []
                self.columns = columns or []
        return [Result(True, [{'id': 1}], ['id']) for _ in queries]

    def batch_compare_queries(self, comparisons):
        class Result:
            def __init__(self, res):
                self.res = res
        return [Result(1) for _ in comparisons]


class MockModelCallable:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def __call__(self, x):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


class TestText2SQLPipeline:

    class MockModel:
        def __init__(self, return_val=None):
            self.return_val = return_val

        def share(self): return self
        def prompt(self, system): return self
        def formatter(self, fmt): return self

        def __call__(self, x, **kwargs):
            if callable(self.return_val):
                return self.return_val(x)
            return self.return_val

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()
        self.db_manager = MockDatabaseManager()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_text2sql_full_pipeline(self):
        sql_forge_response = '```sql\nSELECT * FROM users\n```'
        intent_response = ('[QUESTION-START] Show all users [QUESTION-END]\n'
                           '[EXTERNAL-KNOWLEDGE-START]none[EXTERNAL-KNOWLEDGE-END]')
        auditor_response = 'Yes'
        reasoning_response = 'Step 1: Identify the table. Step 2: Select all columns.'
        effort_response = '```sql\nSELECT * FROM users\n```'

        responses = [
            sql_forge_response,  # SQLForge (1)
            intent_response, intent_response, intent_response, intent_response,
            intent_response,  # SQLIntentSynthesizer (5)
            auditor_response,  # TSQLSemanticAuditor (1)
            reasoning_response, reasoning_response, reasoning_response,  # SQLReasoningTracer (3)
            effort_response, effort_response, effort_response, effort_response, effort_response,
            effort_response, effort_response, effort_response, effort_response, effort_response,  # SQLEffortRanker (10)
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = text2sql_synthetic_ppl(
            model=mock_model,
            database_manager=self.db_manager,
            output_num=1,
            num_generations=3,
            input_query_num=5,
            output_format=None,
        )
        data = [{'db_id': 'test_db'}]
        res = ppl(data)

        assert len(res) == 1
        assert 'SQL' in res[0]
        assert 'intent' in res[0] or 'question' in res[0]
        assert 'cot_reasoning' in res[0]
        assert 'sql_component_difficulty' in res[0]
        assert 'sql_execution_difficulty' in res[0]

    def test_text2sql_pipeline_with_embedding(self):
        sql_forge_response = '```sql\nSELECT id FROM users\n```'
        intent_response = ('[QUESTION-START] Get user ids [QUESTION-END]\n'
                           '[EXTERNAL-KNOWLEDGE-START]none[EXTERNAL-KNOWLEDGE-END]')
        auditor_response = 'Yes'
        reasoning_response = 'Step 1: Identify the users table. Step 2: Select id column.'
        effort_response = '```sql\nSELECT id FROM users\n```'

        responses = [
            sql_forge_response,
            intent_response, intent_response, intent_response, intent_response,
            intent_response,
            auditor_response,
            reasoning_response, reasoning_response, reasoning_response,
            effort_response, effort_response, effort_response, effort_response, effort_response,
            effort_response, effort_response, effort_response, effort_response, effort_response,
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        # Embedding model is optional, pass None
        ppl = text2sql_synthetic_ppl(
            model=mock_model,
            database_manager=self.db_manager,
            embedding_model=None,
            output_num=1,
            num_generations=2,
            input_query_num=5,
            output_format=None,
        )
        data = [{'db_id': 'test_db'}]
        res = ppl(data)

        assert len(res) == 1
        assert 'SQL' in res[0]

    def test_text2sql_enhanced_pipeline(self):
        # SQLQuestionGenerator generates multiple questions from a single LLM response
        # The response should contain output_num questions
        question_response = ('[QUESTION-START] Show all users [QUESTION-END]\n'
                             '[EXTERNAL-KNOWLEDGE-START] none [EXTERNAL-KNOWLEDGE-END]\n'
                             '[QUESTION-START] List user names [QUESTION-END]\n'
                             '[EXTERNAL-KNOWLEDGE-START] none [EXTERNAL-KNOWLEDGE-END]')
        sql_generator_response = '```sql\nSELECT * FROM users\n```'
        reasoning_response = 'Step 1: Identify the table. Step 2: Select all columns.'
        effort_response = '```sql\nSELECT * FROM users\n```'

        # With output_num=2, SQLQuestionGenerator makes 1 call and extracts 2 questions
        # SQLGenerator is called 2 times (once per question)
        # SQLReasoningTracer is called 2 times (once per valid SQL)
        # SQLEffortRanker is called 2 * num_generations = 20 times
        responses = [
            question_response,  # SQLQuestionGenerator: 1 call -> 2 questions
            sql_generator_response, sql_generator_response,  # SQLGenerator: 2 calls
            reasoning_response, reasoning_response,  # SQLReasoningTracer: 2 calls
        ] + [effort_response] * 20  # SQLEffortRanker: 20 calls

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = text2sql_enhanced_ppl(
            model=mock_model,
            database_manager=self.db_manager,
            output_num=2,
            num_generations=10,
            output_format=None,
        )
        data = [{'db_id': 'test_db'}]
        res = ppl(data)

        assert len(res) == 2
        assert 'SQL' in res[0]
        assert 'question' in res[0]
        assert 'cot_reasoning' in res[0]
        assert 'sql_component_difficulty' in res[0]
        assert 'sql_execution_difficulty' in res[0]

    def test_text2sql_enhanced_pipeline_with_format(self):
        question_response = '[QUESTION-START] Get user ids [QUESTION-END]'
        sql_generator_response = '```sql\nSELECT id FROM users\n```'
        reasoning_response = 'Step 1: Identify users table. Step 2: Select id column.'
        effort_response = '```sql\nSELECT id FROM users\n```'

        responses = [
            question_response,
            sql_generator_response,
            reasoning_response,
        ] + [effort_response] * 10

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = text2sql_enhanced_ppl(
            model=mock_model,
            database_manager=self.db_manager,
            embedding_model=None,
            output_num=1,
            num_generations=10,
            output_format='alpaca',
        )
        data = [{'db_id': 'test_db'}]
        res = ppl(data)

        assert len(res) == 1
        assert 'instruction' in res[0]
        assert 'output' in res[0]
