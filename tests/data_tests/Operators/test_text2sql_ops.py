import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data.operators import text2sql_ops

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
        # queries: list of (db_id, sql)
        class Result:
            def __init__(self, success):
                self.success = success
        return [Result(True) for _ in queries]

    def batch_execute_queries(self, queries):
        # queries: list of (db_id, sql)
        class Result:
            def __init__(self, success, data=None, columns=None):
                self.success = success
                self.data = data or []
                self.columns = columns or []
        return [Result(True, [{'id': 1}], ['id']) for _ in queries]

    def batch_compare_queries(self, comparisons):
        # comparisons: list of (db_id, predicted_sql, ground_truth)
        class Result:
            def __init__(self, res):
                self.res = res
        return [Result(1) for _ in comparisons]

class MockModel:
    def __init__(self, return_val=None):
        self.return_val = return_val

    def share(self): return self
    def prompt(self, system): return self
    def formatter(self, fmt): return self

    def __call__(self, x, **kwargs):
        if isinstance(x, list):
            return [self.return_val] * len(x)
        return self.return_val

class TestText2SQLOperators:

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

    def test_sql_generator(self):
        mock_model = MockModel(return_val='```sql SELECT * FROM users ```')
        op = text2sql_ops.SQLForge(
            model=mock_model,
            database_manager=self.db_manager,
            output_num=1,
            _save_data=False
        )
        data = {'db_id': 'test_db'}
        res = op([data])
        assert len(res) == 1
        assert res[0]['SQL'] == 'SELECT * FROM users'
        assert res[0]['db_id'] == 'test_db'

    def test_sql_generator_preserves_fields(self):
        mock_model = MockModel(return_val='```sql SELECT * FROM users ```')
        op = text2sql_ops.SQLGenerator(
            model=mock_model,
            database_manager=self.db_manager,
            output_num=1,
            _save_data=False
        )
        data = {
            'db_id': 'test_db',
            'question': 'Show all users',
            'prompt': 'Schema: CREATE TABLE users...',
            'evidence': 'none'
        }
        res = op([data])
        assert len(res) == 1
        assert res[0]['SQL'] == 'SELECT * FROM users'
        assert res[0]['db_id'] == 'test_db'
        assert res[0]['question'] == 'Show all users'
        assert res[0]['prompt'] == 'Schema: CREATE TABLE users...'
        assert res[0]['evidence'] == 'none'

    def test_sql_executability_filter(self):
        op = text2sql_ops.SQLRuntimeSieve(
            database_manager=self.db_manager,
            _save_data=False
        )
        # Valid SQL
        data_valid = {'SQL': 'SELECT * FROM users', 'db_id': 'test_db'}
        res_valid = op([data_valid])
        assert len(res_valid) == 1
        assert res_valid[0]['SQL'] == 'SELECT * FROM users'

        # Invalid SQL (not starting with SELECT/WITH)
        data_invalid = {'SQL': 'UPDATE users SET name="bob"', 'db_id': 'test_db'}
        res_invalid = op([data_invalid])
        assert len(res_invalid) == 0

    def test_text2sql_question_generator(self):
        mock_model = MockModel(return_val='[QUESTION-START] Who are the users? [QUESTION-END]\n'
                                          '[EXTERNAL-KNOWLEDGE-START] none [EXTERNAL-KNOWLEDGE-END]')
        op = text2sql_ops.SQLIntentSynthesizer(
            model=mock_model,
            database_manager=self.db_manager,
            input_query_num=1,
            input_intent_key='question',
            _save_data=False
        )
        data = {'SQL': 'SELECT * FROM users', 'db_id': 'test_db'}
        res = op([data])
        assert res[0]['question'] == 'Who are the users?'
        assert res[0]['evidence'] == 'none'

    def test_text2sql_correspondence_filter(self):
        mock_model = MockModel(return_val='Yes')
        op = text2sql_ops.TSQLSemanticAuditor(
            model=mock_model,
            database_manager=self.db_manager,
            _save_data=False
        )
        data = {'SQL': 'SELECT * FROM users', 'question': 'Show me all users', 'db_id': 'test_db'}
        res = op([data])
        assert len(res) == 1
        assert res[0]['question'] == 'Show me all users'

        # Mock model returning No
        mock_model.return_val = 'No'
        res_no = op([data])
        assert len(res_no) == 0

    def test_text2sql_prompt_generator(self):
        op = text2sql_ops.SQLContextAssembler(
            database_manager=self.db_manager,
            input_intent_key='question',
            _save_data=False
        )
        data = {'question': 'Show all users', 'db_id': 'test_db', 'evidence': 'none'}
        res = op([data])
        assert 'Database Schema:' in res[0]['prompt']
        assert 'Intent: Show all users' in res[0]['prompt']

    def test_text2sql_cot_generator(self):
        mock_model = MockModel(return_val='Step 1: Select all columns from users table.')
        op = text2sql_ops.SQLReasoningTracer(
            model=mock_model,
            database_manager=self.db_manager,
            output_num=2,
            _save_data=False
        )
        data = {'SQL': 'SELECT * FROM users', 'question': 'Show all users', 'db_id': 'test_db'}
        res = op([data])
        assert len(res[0]['cot_responses']) == 2
        assert res[0]['cot_responses'][0] == 'Step 1: Select all columns from users table.'

    def test_text2sql_cot_voting_generator(self):
        op = text2sql_ops.SQLConsensusUnifier(
            database_manager=self.db_manager,
            _save_data=False
        )
        cot_responses = [
            'Reasoning 1... ```sql SELECT * FROM users ```',
            'Reasoning 2... ```sql SELECT id FROM users ```'
        ]
        data = {'cot_responses': cot_responses, 'db_id': 'test_db'}
        res = op([data])
        assert 'cot_reasoning' in res[0]
        assert res[0]['SQL'] in ['SELECT * FROM users', 'SELECT id FROM users']

    def test_sql_component_classifier(self):
        op = text2sql_ops.SQLSyntaxProfiler(_save_data=False)
        data = {'SQL': 'SELECT name FROM users WHERE id > 1 GROUP BY name HAVING count(*) > 1 ORDER BY name LIMIT 1'}
        res = op([data])
        # Based on EvalHardnessLite logic in SQL_EvalHardness.py
        assert 'sql_component_difficulty' in res[0]
        assert res[0]['sql_component_difficulty'] in ['easy', 'medium', 'hard', 'extra']

    def test_sql_execution_classifier(self):
        mock_model = MockModel(return_val='```sql SELECT * FROM users ```')
        op = text2sql_ops.SQLEffortRanker(
            model=mock_model,
            database_manager=self.db_manager,
            num_generations=5,
            _save_data=False
        )
        data = {'SQL': 'SELECT * FROM users', 'prompt': 'dummy prompt', 'db_id': 'test_db'}
        res = op([data])
        assert 'sql_execution_difficulty' in res[0]
        # Since MockModel always returns the same SQL which matches ground_truth,
        # score will be 10/10 (bumped from 5), which should be 'easy' given the default labels
        # thresholds [2, 5, 9], labels ['extra', 'hard', 'medium', 'easy']
        # score 10 > 9 -> labels[-1] -> 'easy'
        assert res[0]['sql_execution_difficulty'] == 'easy'

    def test_text2sql_to_sft_formatter_alpaca(self):
        '''测试 Text2SQLToSFTFormatter - alpaca 格式'''
        op = text2sql_ops.Text2SQLToSFTFormatter(format_type='alpaca')
        data = {
            'output': {
                'prompt': 'Database Schema:\nCREATE TABLE users...',
                'question': 'Show all users',
                'SQL': 'SELECT * FROM users',
                'cot_reasoning': 'Step 1: Identify table...'
            }
        }
        res = op([data])
        assert len(res) == 1
        assert res[0]['instruction'] == op.DEFAULT_INSTRUCTION
        assert res[0]['input'] == 'Database Schema:\nCREATE TABLE users...'
        assert res[0]['output'] == 'SELECT * FROM users'

    def test_text2sql_to_sft_formatter_cot(self):
        '''测试 Text2SQLToSFTFormatter - cot 格式（默认）'''
        op = text2sql_ops.Text2SQLToSFTFormatter(format_type='cot')
        data = {
            'output': {
                'prompt': 'Database Schema:\nCREATE TABLE users...',
                'question': 'Show all users',
                'SQL': 'SELECT * FROM users',
                'cot_reasoning': 'Step 1: Identify table\nStep 2: Select all'
            }
        }
        res = op([data])
        assert len(res) == 1
        assert res[0]['instruction'] == 'Database Schema:\nCREATE TABLE users...'
        assert res[0]['input'] == ''
        assert 'Step 1: Identify table' in res[0]['output']
        assert 'SELECT * FROM users' in res[0]['output']

    def test_text2sql_to_sft_formatter_flat_structure(self):
        op = text2sql_ops.Text2SQLToSFTFormatter(format_type='cot')
        data = {
            'prompt': 'Database Schema:\nCREATE TABLE orders...',
            'question': 'Count orders',
            'SQL': 'SELECT COUNT(*) FROM orders',
            'cot_reasoning': 'Step 1: Count records'
        }
        res = op([data])
        assert len(res) == 1
        assert res[0]['instruction'] == 'Database Schema:\nCREATE TABLE orders...'
        assert res[0]['output'] == '<think>\nStep 1: Count records\n</think>\n\nSELECT COUNT(*) FROM orders'

    def test_text2sql_to_sft_formatter_missing_sql(self):
        op = text2sql_ops.Text2SQLToSFTFormatter(format_type='cot')
        data = {
            'output': {
                'prompt': 'Some prompt',
                'question': 'Some question'
            }
        }
        res = op([data])
        assert len(res) == 0

    def test_text2sql_to_sft_formatter_missing_cot(self):
        '''测试 Text2SQLToSFTFormatter - 缺少 CoT 时只输出 SQL'''
        op = text2sql_ops.Text2SQLToSFTFormatter(format_type='cot')
        data = {
            'output': {
                'prompt': 'Database Schema:\nCREATE TABLE users...',
                'question': 'Show all users',
                'SQL': 'SELECT * FROM users'
            }
        }
        res = op([data])
        assert len(res) == 1
        assert res[0]['output'] == 'SELECT * FROM users'

    def test_sql_question_generator(self):
        '''测试 SQLQuestionGenerator 根据 schema 生成问题'''
        mock_model = MockModel(
            return_val=(
                '[QUESTION-START]How many users are there?[QUESTION-END]\n'
                '[EXTERNAL-KNOWLEDGE-START]Count all records[EXTERNAL-KNOWLEDGE-END]'
            )
        )
        op = text2sql_ops.SQLQuestionGenerator(
            model=mock_model,
            database_manager=self.db_manager,
            output_num=1,
            target_complexity='hard',
            _save_data=False
        )
        data = {'db_id': 'test_db'}
        res = op([data])
        assert len(res) == 1
        assert res[0]['question'] == 'How many users are there?'
        assert res[0]['evidence'] == 'Count all records'
