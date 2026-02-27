from ..base_data import data_register
import random
import re
import math
from collections import defaultdict
from lazyllm.thirdparty import pandas as pd
from tqdm import tqdm
from lazyllm import LOG
from .sql_evalhardness import EvalHardness, EvalHardnessLite, Schema


Text2SQLOps = data_register.new_group('text2sql_ops')


def _stringify_statements(statements, max_lines=None):
    if statements is None:
        return ''
    if isinstance(statements, str):
        text = statements
    elif isinstance(statements, (list, tuple)):
        text = '\n'.join([str(x) for x in statements if x is not None])
    else:
        text = str(statements)
    lines = [ln for ln in text.splitlines() if ln.strip() != '']
    if max_lines is not None:
        lines = lines[:max_lines]
    return '\n'.join(lines)


def _default_build_prompt(create_statements, insert_statements, db_engine):
    complexity = random.choice(['easy', 'medium', 'hard'])

    schema_text = _stringify_statements(create_statements, max_lines=400)
    sample_text = _stringify_statements(insert_statements, max_lines=200)

    instruction = (
        f'You are a Text2SQL data generation assistant.\n'
        f'Database engine: {db_engine}\n\n'
        f'DDL:\n{schema_text}\n\n'
        f'Sample data (optional):\n{sample_text}\n\n'
        f'Task: Generate ONE {complexity} SQL query that is executable for this database.\n'
        f'Requirements:\n'
        f'- Return ONLY one SQL query inside a Markdown code block: ```sql ... ```.\n'
        f'- Do NOT include explanations.\n'
    )
    return instruction, complexity


def _parse_sql_response(response):
    if response is None:
        return ''
    if not isinstance(response, str):
        response = str(response)
    response = response.strip()
    if not response:
        return ''
    blocks = re.findall(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    matches = re.findall(r'\bselect\b.*?\bfrom\b', response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return ''


def _result_to_signature(result):
    if not result.success or not result.data:
        return None
    columns = result.columns or []
    if not columns and result.data:
        columns = list(result.data[0].keys())
    return frozenset(
        tuple(row.get(c) for c in columns) for row in result.data
    )


def _tie_break(candidates, tie_breaker='shortest_sql'):
    if not candidates:
        raise ValueError('tie_break candidates empty')
    if tie_breaker == 'random':
        return random.choice(candidates)
    return min(candidates, key=lambda x: len(x.get('sql') or ''))


def _vote_select(candidates, tie_breaker='shortest_sql'):
    valid = [c for c in candidates if c.get('is_valid')]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    buckets = defaultdict(list)
    for c in valid:
        buckets[c['signature']].append(c)

    best_bucket = max(buckets.values(), key=len)
    if len(best_bucket) >= 2:
        return random.choice(best_bucket)
    return _tie_break(valid, tie_breaker)


class SQLForge(Text2SQLOps):
    def __init__(self, model=None, database_manager=None, output_num=300,
                 prompt_template=None, system_prompt=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.database_manager = database_manager
        self.output_num = output_num
        self.prompt_template = prompt_template
        sys_prompt = system_prompt or (
            'You are a SQL generator for Text2SQL tasks.\n'
            'Return ONLY one SQL query inside a Markdown code block: ```sql ... ```.\n'
        )
        self.model = model.share().prompt(sys_prompt) if model else None

    def _build_prompt(self, create_statements, insert_statements, db_engine):
        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            built = template.build_prompt(
                insert_statements=insert_statements,
                create_statements=create_statements,
                db_engine=db_engine,
            )
            if isinstance(built, tuple) and len(built) >= 2:
                return str(built[0]), str(built[1])
            return str(built), 'unknown'
        return _default_build_prompt(create_statements, insert_statements, db_engine)

    def _validate_manager(self):
        if self.model is None:
            raise ValueError('model is required')
        if self.database_manager is None:
            raise ValueError('database_manager is required')
        if not hasattr(self.database_manager, 'list_databases'):
            raise ValueError('database_manager.list_databases is required')
        if not hasattr(self.database_manager, 'get_create_statements_and_insert_statements'):
            raise ValueError('database_manager.get_create_statements_and_insert_statements is required')

    def forward(self, data, output_sql_key='SQL', output_db_id_key='db_id',
                output_complexity_type_key='sql_complexity_type', **kwargs):
        assert isinstance(data, dict)
        self._validate_manager()

        db_engine = getattr(self.database_manager, 'db_type', 'unknown')
        db_id_in_data = data.get(output_db_id_key)
        if db_id_in_data:
            db_names = [db_id_in_data]
        else:
            db_names = self.database_manager.list_databases() or []

        if not db_names:
            LOG.warning('No databases found in database_manager.list_databases()')
            return []

        prompts, db_ids, complexity_types = self._collect_prompts(db_names, db_engine)

        responses = []
        for p in prompts:
            try:
                responses.append(self.model(p))
            except Exception as e:
                LOG.error(f'Failed to generate SQL: {e}')
                responses.append('')

        return [
            {
                output_db_id_key: db_id,
                output_sql_key: _parse_sql_response(resp),
                output_complexity_type_key: complexity,
            }
            for db_id, resp, complexity in zip(db_ids, responses, complexity_types)
        ]

    def _collect_prompts(self, db_names, db_engine):
        prompts = []
        db_ids = []
        complexity_types = []

        LOG.info(f'Generating {self.output_num} SQLs for each database')
        for db_name in tqdm(db_names, desc='Processing Databases'):
            create_statements, insert_statements = self.database_manager.get_create_statements_and_insert_statements(
                db_name
            )
            for _ in range(int(self.output_num)):
                prompt, complexity = self._build_prompt(create_statements, insert_statements, db_engine=db_engine)
                prompts.append(prompt)
                db_ids.append(db_name)
                complexity_types.append(complexity)
        return prompts, db_ids, complexity_types


class SQLRuntimeSieve(Text2SQLOps):
    def __init__(self, database_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.database_manager = database_manager

    def filter_select_sql(self, sql):
        if not isinstance(sql, str):
            return False
        sql_wo_comments = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        sql_wo_comments = re.sub(r'--.*', '', sql_wo_comments)
        sql_wo_comments = sql_wo_comments.strip()

        if sql_wo_comments.lower().startswith('select') or \
           sql_wo_comments.lower().startswith('with'):
            return True
        return False

    def forward(self, data, input_sql_key='SQL', input_db_id_key='db_id', **kwargs):
        assert isinstance(data, dict)
        if self.database_manager is None:
            LOG.error('database_manager is required for SQLExecutabilityFilter')
            return data

        sql = data.get(input_sql_key)
        db_id = data.get(input_db_id_key)

        if not self.filter_select_sql(sql):
            return []

        if not self.database_manager.database_exists(db_id):
            LOG.warning(f'Database {db_id} not found in registry, please check the database folder')
            return []

        try:
            execution_results = self.database_manager.batch_explain_queries([(db_id, sql)])
            if execution_results and execution_results[0].success:
                return data
        except Exception as e:
            LOG.error(f'Error during explain_query: {e}')

        return []


class SQLIntentSynthesizer(Text2SQLOps):
    def __init__(self, model=None, embedding_model=None, database_manager=None,
                 input_query_num=5, prompt_template=None, system_prompt=None,
                 input_intent_key='intent', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.embedding_model = embedding_model
        self.database_manager = database_manager
        self.question_candidates_num = int(input_query_num)
        self.prompt_template = prompt_template
        self.input_intent_key = input_intent_key
        sys_prompt = system_prompt or 'You are a helpful assistant.'
        self.model = model.share().prompt(sys_prompt) if model else None

    @staticmethod
    def _is_non_empty_text(x):
        return isinstance(x, str) and x.strip() != ''

    def extract_column_descriptions(self, create_statements):
        column_name2column_desc = {}
        pattern = r'"(\w+)"\s+\w+\s*/\*\s*(.*?)\s*\*/'
        if not create_statements:
            return column_name2column_desc
        for create_statement in create_statements:
            for column_name, description in re.findall(pattern, str(create_statement)):
                col = str(column_name).lower()
                if col not in column_name2column_desc:
                    column_name2column_desc[col] = str(description)
        return column_name2column_desc

    def parse_llm_response(self, response):
        if not isinstance(response, str):
            LOG.warning(f'Invalid response type: {type(response)}, expected str. Response: {response}')
            return None

        question_pattern = re.compile(r'\[QUESTION-START\](.*?)\[QUESTION-END\]', re.DOTALL)
        external_knowledge_pattern = re.compile(
            r'\[EXTERNAL-KNOWLEDGE-START\](.*?)\[EXTERNAL-KNOWLEDGE-END\]', re.DOTALL
        )

        question_match = question_pattern.search(response)
        external_knowledge_match = external_knowledge_pattern.search(response)

        question_content = question_match.group(1).strip() if question_match else ''
        external_knowledge_content = external_knowledge_match.group(1).strip() if external_knowledge_match else ''

        if question_content == '':
            return None
        return {'question': question_content, 'external_knowledge': external_knowledge_content}

    @staticmethod
    def _cosine_distance(a, b):
        if not a or not b:
            return 1.0
        n = min(len(a), len(b))
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            x = float(a[i])
            y = float(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        denom = math.sqrt(na) * math.sqrt(nb)
        if denom == 0.0:
            return 1.0
        return 1.0 - (dot / denom)

    def _select_best_question(self, question_candidates, start_idx, embeddings):
        if not question_candidates:
            return None
        if len(question_candidates) == 1:
            return question_candidates[0]
        if embeddings is None or start_idx < 0:
            return random.sample(question_candidates, 1)[0]

        end_idx = start_idx + len(question_candidates)
        if end_idx > len(embeddings):
            return random.sample(question_candidates, 1)[0]

        candidate_embeddings = embeddings[start_idx:end_idx]
        distance_sums = []
        for i in range(len(candidate_embeddings)):
            s = 0.0
            for j in range(len(candidate_embeddings)):
                if i == j:
                    continue
                s += self._cosine_distance(candidate_embeddings[i], candidate_embeddings[j])
            distance_sums.append(s)
        min_index = min(range(len(distance_sums)), key=distance_sums.__getitem__)
        return question_candidates[min_index]

    def _build_prompt(self, sql, db_id, db_id2column_info, db_engine):
        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            built = template.build_prompt(sql, db_id, db_id2column_info, db_engine)
            if isinstance(built, tuple) and len(built) >= 2:
                return str(built[0]), str(built[1])
            return str(built), 'unknown'

        column_info = db_id2column_info.get(db_id, {})
        column_info_text = '\n'.join([f'- {k}: {v}' for k, v in list(column_info.items())[:200]])
        prompt = (
            f'You are a Text2SQL intent synthesizer.\n'
            f'Database engine: {db_engine}\n'
            f'db_id: {db_id}\n\n'
            f'Given a SQL query, generate a natural language intent that matches it.\n'
            f'If helpful, you may use the following column descriptions:\n{column_info_text}\n\n'
            f'Output format:\n'
            f'[INTENT-START] ... [INTENT-END]\n'
            f'[EXTERNAL-KNOWLEDGE-START] ... [EXTERNAL-KNOWLEDGE-END]\n\n'
            f'SQL:\n{sql}\n'
        )
        return prompt, 'default'

    def _generate_embeddings(self, texts):
        if not texts:
            return []
        emb = self.embedding_model
        if emb is None:
            return None
        try:
            if hasattr(emb, 'generate_embedding_from_input'):
                vectors = emb.generate_embedding_from_input(texts)
            elif callable(emb):
                vectors = emb(texts)
            else:
                return None
            if not isinstance(vectors, list):
                return None
            return vectors
        except Exception as e:
            LOG.warning(f'Embedding generation failed: {e}')
            return None

    def _validate_generator_manager(self):
        if self.model is None:
            raise ValueError('model is required')
        if self.database_manager is None:
            raise ValueError('database_manager is required')
        if not hasattr(self.database_manager, 'get_create_statements_and_insert_statements'):
            raise ValueError('database_manager.get_create_statements_and_insert_statements is required')

    def forward(self, data, input_sql_key='SQL', input_db_id_key='db_id',
                output_intent_key=None, output_evidence_key='evidence', **kwargs):
        assert isinstance(data, dict)
        self._validate_generator_manager()

        if output_intent_key is None:
            output_intent_key = self.input_intent_key

        if self._is_non_empty_text(data.get(self.input_intent_key)):
            return data

        db_engine = getattr(self.database_manager, 'db_type', 'unknown')
        sql = data.get(input_sql_key, '')
        db_id = data.get(input_db_id_key, '')

        try:
            create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
            column_info = self.extract_column_descriptions(create_statements)
        except Exception as e:
            LOG.warning(f'Failed to extract schema for db_id={db_id}: {e}')
            column_info = {}

        prompt, question_type = self._build_prompt(str(sql), str(db_id), {db_id: column_info}, db_engine)
        data['question_type'] = question_type

        responses = []
        for _ in range(self.question_candidates_num):
            try:
                responses.append(self.model(prompt))
            except Exception as e:
                LOG.error(f'Failed to generate question: {e}')
                responses.append('')

        candidates = []
        embedding_texts = []
        for resp in responses:
            parsed = self.parse_llm_response(resp)
            if parsed:
                candidates.append(parsed)
                text = f'{parsed.get("external_knowledge", "")} {parsed.get("question", "")}'.strip()
                embedding_texts.append(text)

        embeddings = self._generate_embeddings(embedding_texts) if embedding_texts else None
        best = self._select_best_question(candidates, 0, embeddings)

        if best is not None:
            data[output_intent_key] = best.get('question', '')
            data[output_evidence_key] = best.get('external_knowledge', '')

        return data

class TSQLSemanticAuditor(Text2SQLOps):
    def __init__(self, model=None, database_manager=None, prompt_template=None,
                 system_prompt=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.database_manager = database_manager
        self.prompt_template = prompt_template
        sys_prompt = system_prompt or (
            'You are an expert in SQL and database analysis.\n'
            'Your task is to determine if a given SQL query correctly answers a natural language '
            'question based on the provided database schema.\n'
            'Respond ONLY with "Yes" if the SQL is correct and "No" otherwise.'
        )
        self.model = model.share().prompt(sys_prompt) if model else None

    def _parse_consistency_response(self, response):
        if not isinstance(response, str):
            return False
        response = response.strip().lower()
        if 'yes' in response:
            return True
        return False

    def _build_prompt(self, question, sql, db_details):
        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            return str(template.build_prompt(question, sql, db_details))

        return (
            f'Database Schema:\n{db_details}\n\n'
            f'Question: {question}\n\n'
            f'SQL Query: {sql}\n\n'
            f'Does the SQL query correctly answer the question according to the schema? (Yes/No)'
        )

    def forward(self, data, input_sql_key='SQL', input_db_id_key='db_id',
                input_question_key='question', input_evidence_key='evidence', **kwargs):
        assert isinstance(data, dict)
        if self.model is None:
            raise ValueError('model is required')
        if self.database_manager is None:
            raise ValueError('database_manager is required')

        sql = data.get(input_sql_key)
        question = data.get(input_question_key)
        evidence = data.get(input_evidence_key, '')
        db_id = data.get(input_db_id_key)

        if not question or str(question).strip() == '':
            return []

        if evidence:
            question = f'{question}\n{evidence}'

        try:
            create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
            db_details = '\n\n'.join([str(s) for s in create_statements])
            prompt = self._build_prompt(str(question), str(sql), db_details)
            response = self.model(prompt)
            if self._parse_consistency_response(response):
                return data
        except Exception as e:
            LOG.warning(f'Failed to check correspondence: {e}')

        return []


class SQLContextAssembler(Text2SQLOps):
    def __init__(self, database_manager=None, prompt_template=None, **kwargs):
        super().__init__(**kwargs)
        self.database_manager = database_manager
        self.prompt_template = prompt_template

    def get_create_statements_and_insert_statements(self, db_id):
        return self.database_manager.get_create_statements_and_insert_statements(db_id)

    def _build_prompt(self, db_details, intent, evidence, db_engine):
        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            return str(template.build_prompt(
                db_details=db_details,
                intent=intent,
                evidence=evidence,
                db_engine=db_engine
            ))

        return (
            f'Database Schema:\n{db_details}\n\n'
            f'Intent: {intent}\n'
            f'Evidence: {evidence}\n'
            f'Generate a SQL query for {db_engine}.'
        )

    def forward(self, data, input_intent_key='intent', input_db_id_key='db_id',
                input_evidence_key='evidence', output_prompt_key='prompt', **kwargs):
        assert isinstance(data, dict)
        if self.database_manager is None:
            raise ValueError('database_manager is required')

        db_engine = getattr(self.database_manager, 'db_type', 'unknown')
        db_id = data.get(input_db_id_key)
        intent = data.get(input_intent_key)
        evidence = data.get(input_evidence_key, '')

        if not db_id or not intent:
            LOG.warning(f'Missing db_id or intent for item: {data}')
            data[output_prompt_key] = ''
            return data

        db_id = str(db_id).replace('\n', '').replace('\r', '').strip()

        try:
            if hasattr(self.database_manager, 'get_db_details'):
                db_details = self.database_manager.get_db_details(db_id)
            else:
                create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
                db_details = '\n\n'.join([str(s) for s in create_statements])

            prompt = self._build_prompt(
                db_details=db_details,
                intent=intent,
                evidence=evidence,
                db_engine=db_engine
            )
            data[output_prompt_key] = prompt
        except Exception as e:
            LOG.error(f'Failed to generate context for db_id={db_id}: {e}')
            data[output_prompt_key] = ''

        return data


class SQLReasoningTracer(Text2SQLOps):
    def __init__(self, model=None, database_manager=None, prompt_template=None,
                 output_num=3, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.database_manager = database_manager
        self.prompt_template = prompt_template
        self.output_num = int(output_num)
        if self.output_num < 1:
            raise ValueError('output_num must be >= 1')
        sys_prompt = 'You are a database expert. Please generate a step-by-step reasoning ' \
                     '(Chain of Thought) and the final SQL.'
        self.model = model.share().prompt(sys_prompt) if model else None

    def _build_prompt(self, item, schema_str):
        intent = item.get(self.input_intent_key)
        gold_sql = item.get(self.input_sql_key)
        evidence = item.get(self.input_evidence_key, '')

        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            return template.build_prompt(schema_str, intent, gold_sql, evidence)

        return (
            f'Database Schema:\n{schema_str}\n\n'
            f'Intent: {intent}\n'
            f'Evidence: {evidence}\n'
            f'Target SQL: {gold_sql}\n\n'
            f'Please provide a detailed step-by-step reasoning that leads to the correct SQL query.'
        )

    def forward(self, data, input_sql_key='SQL', input_intent_key='intent',
                input_db_id_key='db_id', input_evidence_key='evidence',
                output_cot_key='cot_responses', **kwargs):
        assert isinstance(data, dict)
        self._validate_manager()

        self.input_intent_key = input_intent_key
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.input_evidence_key = input_evidence_key

        db_id = data.get(input_db_id_key)
        if not db_id:
            LOG.warning('Missing db_id for reasoning tracing')
            return data

        try:
            create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
            schema_str = '\n\n'.join([str(s) for s in create_statements])
            prompt = self._build_prompt(data, schema_str)

            responses = []
            for _ in range(self.output_num):
                try:
                    responses.append(self.model(prompt))
                except Exception as e:
                    LOG.error(f'Failed to generate reasoning trace: {e}')
                    responses.append('')
            data[output_cot_key] = responses
        except Exception as e:
            LOG.error(f'Error during reasoning tracing for db_id={db_id}: {e}')
            data[output_cot_key] = []

        return data

    def _validate_manager(self):
        if self.model is None:
            raise ValueError('model is required')
        if self.database_manager is None:
            raise ValueError('database_manager is required')

class SQLConsensusUnifier(Text2SQLOps):
    def __init__(self, database_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.database_manager = database_manager
        self.tie_breaker = 'shortest_sql'

    def forward(
        self,
        data,
        input_cot_responses_key='cot_responses',
        input_db_id_key='db_id',
        output_cot_key='cot_reasoning',
        output_sql_key='SQL',
        **kwargs,
    ):
        assert isinstance(data, dict)
        if self.database_manager is None:
            raise ValueError('database_manager is required')

        cot_responses = data.get(input_cot_responses_key, [])
        if not isinstance(cot_responses, list) or not cot_responses:
            data[output_cot_key] = ''
            return data

        db_id = data.get(input_db_id_key)
        if not db_id:
            data[output_cot_key] = ''
            return data

        candidates = []
        queries = []
        for resp in cot_responses:
            sql = _parse_sql_response(resp)
            if sql:
                queries.append((str(db_id).strip(), sql))
                candidates.append({'cot': resp, 'sql': sql})

        if not queries:
            data[output_cot_key] = ''
            return data

        try:
            query_results = self.database_manager.batch_execute_queries(queries)
            for cand, result in zip(candidates, query_results):
                cand['signature'] = _result_to_signature(result)
                cand['is_valid'] = result.success if hasattr(result, 'success') else False
        except Exception as e:
            LOG.error(f'Failed to execute queries for voting: {e}')

        best = _vote_select(candidates, self.tie_breaker)
        if best:
            data[output_cot_key] = best.get('cot', '')
            data[output_sql_key] = best.get('sql', '')
        else:
            data[output_cot_key] = ''

        return data


class SQLSyntaxProfiler(Text2SQLOps):
    def __init__(self, difficulty_thresholds=None, difficulty_labels=None, **kwargs):
        super().__init__(**kwargs)
        if difficulty_thresholds is None:
            difficulty_thresholds = [2, 4, 6]
        if difficulty_labels is None:
            difficulty_labels = ['easy', 'medium', 'hard', 'extra']

        self.difficulty_config = {
            'thresholds': difficulty_thresholds,
            'labels': difficulty_labels,
        }
        if len(self.difficulty_config['thresholds']) != len(self.difficulty_config['labels']) - 1:
            raise ValueError('Thresholds and labels configuration mismatch')

    def eval_component_hardness(self, sql, schema):
        evaluator = EvalHardness(Schema(schema), sql)
        return evaluator.run()

    def eval_hardness_lite(self, sql):
        evaluator = EvalHardnessLite(str(sql), self.difficulty_config)
        return evaluator.run()

    def forward(self, data, input_sql_key='SQL',
                output_difficulty_key='sql_component_difficulty', **kwargs):
        assert isinstance(data, dict)
        sql = data.get(input_sql_key)
        if not sql:
            data[output_difficulty_key] = 'unknown'
            return data
        hardness = self.eval_hardness_lite(str(sql))
        data[output_difficulty_key] = hardness

        return data

class SQLEffortRanker(Text2SQLOps):
    def __init__(self, model=None, database_manager=None, num_generations=10,
                 difficulty_thresholds=None, difficulty_labels=None,
                 system_prompt=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.database_manager = database_manager
        self.num_generations = int(num_generations)
        sys_prompt = system_prompt or (
            'You are a SQL generator. '
            'Return ONLY the SQL query inside a Markdown code block: ```sql ... ```.'
        )
        self.model = model.share().prompt(sys_prompt) if model else None
        if difficulty_thresholds is None:
            difficulty_thresholds = [2, 5, 9]
        if difficulty_labels is None:
            difficulty_labels = ['extra', 'hard', 'medium', 'easy']

        self.difficulty_config = {
            'thresholds': difficulty_thresholds,
            'labels': difficulty_labels,
        }
        self.timeout = 5.0

        if self.num_generations <= self.difficulty_config['thresholds'][-1]:
            nearest_multiple = ((self.difficulty_config['thresholds'][-1] // 5) + 1) * 5
            LOG.warning(f'num_generations is less than the last threshold, will be set to {nearest_multiple}')
            self.num_generations = nearest_multiple
        if len(self.difficulty_config['thresholds']) != len(self.difficulty_config['labels']) - 1:
            raise ValueError('Thresholds and labels configuration mismatch')

    @staticmethod
    def parse_response(response):
        return _parse_sql_response(response)

    @staticmethod
    def _prepare_comparisons(predicted_sqls_list, ground_truth_list, db_ids, idxs):
        comparisons = []
        for predicted_sqls, ground_truth, db_id in zip(predicted_sqls_list, ground_truth_list, db_ids):
            for predicted_sql in predicted_sqls:
                comparisons.append((db_id, predicted_sql, ground_truth))
        return comparisons

    def classify_difficulty(self, score):
        if score == -1:
            return 'gold error'
        thresholds = self.difficulty_config['thresholds']
        labels = self.difficulty_config['labels']
        for i, threshold in enumerate(thresholds):
            if score <= threshold:
                return labels[i]
        return labels[-1]

    def report_statistics(self, inputs, output_difficulty_key):
        difficulties = [item.get(output_difficulty_key) for item in inputs]
        counts = pd.Series(difficulties).value_counts()
        LOG.info('SQL Difficulty Statistics')
        stats = [f'{d.title()}: {counts.get(d, 0)}' for d in ['easy', 'medium', 'hard', 'extra', 'gold error']]
        LOG.info(', '.join(stats))

    def _generate_and_parse_sqls(self, input_prompts):
        prompts = [q for q in input_prompts for _ in range(self.num_generations)]
        responses = []
        try:
            responses = self.model(prompts)
            if isinstance(responses, str):
                responses = [responses]
        except Exception as e:
            LOG.error(f'Generation failed: {e}')
            responses = [''] * len(prompts)

        all_parsed_sqls = []
        for response in responses:
            all_parsed_sqls.append(_parse_sql_response(response) if response else '')
        return all_parsed_sqls

    def forward(self, data, input_db_id_key='db_id', input_sql_key='SQL',
                input_prompt_key='prompt', output_difficulty_key='sql_execution_difficulty',
                **kwargs):
        assert isinstance(data, dict)
        if self.model is None or self.database_manager is None:
            raise ValueError('model and database_manager are required')

        prompt = data.get(input_prompt_key)
        ground_truth = data.get(input_sql_key)
        db_id = data.get(input_db_id_key)

        if not prompt or not ground_truth or not db_id:
            data[output_difficulty_key] = 'unknown'
            return data

        prompts = [prompt] * self.num_generations
        try:
            responses = self.model(prompts)
            if isinstance(responses, str):
                responses = [responses]
        except Exception as e:
            LOG.error(f'Generation failed: {e}')
            responses = [''] * self.num_generations

        parsed_sqls = [_parse_sql_response(r) if r else '' for r in responses]

        comparisons = [(db_id, sql, ground_truth) for sql in parsed_sqls]
        try:
            batch_results = self.database_manager.batch_compare_queries(comparisons)
            cnt_true = sum(1 for res in batch_results if res.res == 1)
            data[output_difficulty_key] = self.classify_difficulty(cnt_true)
        except Exception as e:
            LOG.error(f'Comparison failed: {e}')
            data[output_difficulty_key] = 'gold error'

        return data
