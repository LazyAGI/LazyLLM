from ..base_data import data_register
import random
import re
import os
import math
from collections import defaultdict
from lazyllm.thirdparty import pandas as pd
from lazyllm.thirdparty import nltk
from tqdm import tqdm
from lazyllm import LOG


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


class SQLGenerator(Text2SQLOps):
    def __init__(self, model=None, database_manager=None, generate_num=300,
                 prompt_template=None, system_prompt=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.database_manager = database_manager
        self.generate_num = generate_num
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

    def forward_batch_input(self, inputs, output_sql_key='SQL', output_db_id_key='db_id',
                            output_sql_complexity_key='sql_complexity_type', **kwargs):
        self._validate_manager()

        db_engine = getattr(self.database_manager, 'db_type', 'unknown')
        db_names = self.database_manager.list_databases() or []
        if not db_names:
            LOG.warning('No databases found in database_manager.list_databases()')
            return []

        prompts, db_ids, complexity_types = self._collect_prompts(db_names, db_engine)

        responses = []
        for p in tqdm(prompts, desc='Generating SQL'):
            try:
                responses.append(self.model(p))
            except Exception as e:
                LOG.error(f'Failed to generate SQL: {e}')
                responses.append('')

        return [
            {
                output_db_id_key: db_id,
                output_sql_key: _parse_sql_response(resp),
                output_sql_complexity_key: complexity,
            }
            for db_id, resp, complexity in zip(db_ids, responses, complexity_types)
        ]

    def _collect_prompts(self, db_names, db_engine):
        prompts = []
        db_ids = []
        complexity_types = []

        LOG.info(f'Generating {self.generate_num} SQLs for each database')
        for db_name in tqdm(db_names, desc='Processing Databases'):
            create_statements, insert_statements = self.database_manager.get_create_statements_and_insert_statements(
                db_name
            )
            for _ in range(int(self.generate_num)):
                prompt, complexity = self._build_prompt(create_statements, insert_statements, db_engine=db_engine)
                prompts.append(prompt)
                db_ids.append(db_name)
                complexity_types.append(complexity)
        return prompts, db_ids, complexity_types


class SQLExecutabilityFilter(Text2SQLOps):
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

    def forward_batch_input(self, inputs, input_sql_key='SQL', input_db_id_key='db_id', **kwargs):
        if not inputs:
            return []

        if self.database_manager is None:
            LOG.error('database_manager is required for SQLExecutabilityFilter')
            return inputs

        db_ids = {item.get(input_db_id_key) for item in inputs if item.get(input_db_id_key)}
        for db_id in db_ids:
            if not self.database_manager.database_exists(db_id):
                LOG.warning(f'Database {db_id} not found in registry, please check the database folder')

        LOG.info(f'Start to filter {len(inputs)} SQLs')

        phase1_inputs = [
            item for item in inputs
            if self.filter_select_sql(item.get(input_sql_key))
        ]

        LOG.info(f'Phase 1 completed: {len(phase1_inputs)}/{len(inputs)} SQLs passed initial filter')

        if not phase1_inputs:
            return []

        sql_triples = [
            (item.get(input_db_id_key), item.get(input_sql_key))
            for item in phase1_inputs
        ]

        try:
            execution_results = self.database_manager.batch_explain_queries(sql_triples)
            results = [
                item for item, res in zip(phase1_inputs, execution_results)
                if res.success
            ]
        except Exception as e:
            LOG.error(f'Error during batch_explain_queries: {e}')
            results = []

        LOG.info(f'Filter completed, remaining {len(results)} SQLs out of {len(phase1_inputs)} (phase1)')
        return results


class Text2SQLQuestionGenerator(Text2SQLOps):
    def __init__(self, model=None, embedding_model=None, database_manager=None,
                 question_candidates_num=5, prompt_template=None, system_prompt=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.embedding_model = embedding_model
        self.database_manager = database_manager
        self.question_candidates_num = int(question_candidates_num)
        self.prompt_template = prompt_template
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
            f'You are a Text2SQL question generator.\n'
            f'Database engine: {db_engine}\n'
            f'db_id: {db_id}\n\n'
            f'Given a SQL query, generate a natural language question that matches it.\n'
            f'If helpful, you may use the following column descriptions:\n{column_info_text}\n\n'
            f'Output format:\n'
            f'[QUESTION-START] ... [QUESTION-END]\n'
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

    def forward_batch_input(self, inputs, input_sql_key='SQL', input_db_id_key='db_id',
                            output_question_key='question', output_evidence_key='evidence', **kwargs):
        if not inputs:
            return []
        self._validate_generator_manager()

        results, to_generate = self._split_inputs(inputs, output_question_key)
        if not to_generate:
            return results

        db_engine = getattr(self.database_manager, 'db_type', 'unknown')
        db_id2column_info = self._get_db_column_info(to_generate, input_db_id_key)
        prompts, _ = self._prepare_prompts(
            to_generate, input_sql_key, input_db_id_key, db_id2column_info, db_engine
        )

        responses = self._generate_responses(prompts)
        processed_to_generate = self._process_responses(
            to_generate, responses, output_question_key, output_evidence_key
        )

        return results + processed_to_generate

    def _split_inputs(self, inputs, output_question_key):
        results = []
        to_generate = []
        for item in inputs:
            if self._is_non_empty_text(item.get(output_question_key)):
                results.append(dict(item))
            else:
                to_generate.append(dict(item))
        return results, to_generate

    def _generate_responses(self, prompts):
        responses = []
        for p in tqdm(prompts, desc='Generating questions'):
            try:
                responses.append(self.model(p))
            except Exception as e:
                LOG.error(f'Failed to generate question: {e}')
                responses.append('')
        return responses

    def _process_responses(self, to_generate, responses, output_question_key, output_evidence_key):
        group_size = max(self.question_candidates_num, 1)
        grouped_responses = [responses[i:i + group_size] for i in range(0, len(responses), group_size)]

        question_groups = []
        embedding_texts = []
        for resp_group in grouped_responses:
            candidates = []
            for resp in resp_group:
                parsed = self.parse_llm_response(resp)
                if parsed:
                    candidates.append(parsed)
                    text = f'{parsed.get("external_knowledge", "")} {parsed.get("question", "")}'.strip()
                    embedding_texts.append(text)
            question_groups.append(candidates)

        embeddings = self._generate_embeddings(embedding_texts) if embedding_texts else None
        processed_results = []
        embedding_start_idx = 0
        for item, candidates in zip(to_generate, question_groups):
            best = self._select_best_question(candidates, embedding_start_idx, embeddings)
            if best is not None:
                item[output_question_key] = best.get('question', '')
                item[output_evidence_key] = best.get('external_knowledge', '')
                embedding_start_idx += len(candidates)
            processed_results.append(item)
        return processed_results

    def _get_db_column_info(self, to_generate, input_db_id_key):
        db_ids = sorted({x.get(input_db_id_key) for x in to_generate if self._is_non_empty_text(x.get(input_db_id_key))})
        db_id2column_info = {}
        for db_id in tqdm(db_ids, desc='Extracting database schema'):
            try:
                create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
                db_id2column_info[db_id] = self.extract_column_descriptions(create_statements)
            except Exception as e:
                LOG.warning(f'Failed to extract schema for db_id={db_id}: {e}')
                db_id2column_info[db_id] = {}
        return db_id2column_info

    def _prepare_prompts(self, to_generate, input_sql_key, input_db_id_key,
                         db_id2column_info, db_engine):
        prompts = []
        question_types = []
        for item in tqdm(to_generate, desc='Preparing prompts'):
            sql = item.get(input_sql_key, '')
            db_id = item.get(input_db_id_key, '')
            prompt, question_type = self._build_prompt(str(sql), str(db_id), db_id2column_info, db_engine)
            item['question_type'] = question_type
            for _ in range(max(self.question_candidates_num, 1)):
                prompts.append(prompt)
            question_types.append(question_type)
        return prompts, question_types


class Text2SQLCorrespondenceFilter(Text2SQLOps):
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

    def forward_batch_input(self, inputs, input_sql_key='SQL', input_db_id_key='db_id',
                            input_question_key='question', input_evidence_key='evidence', **kwargs):
        if not inputs:
            return []
        if self.model is None:
            raise ValueError('model is required')
        if self.database_manager is None:
            raise ValueError('database_manager is required')

        prompts, valid_indices = self._prepare_prompts_and_indices(
            inputs, input_sql_key, input_db_id_key, input_question_key, input_evidence_key
        )

        if not prompts:
            return []

        responses = self._generate_responses(prompts)
        return self._filter_results(inputs, valid_indices, responses)

    def _prepare_prompts_and_indices(self, inputs, input_sql_key, input_db_id_key,
                                     input_question_key, input_evidence_key):
        prompts = []
        valid_indices = []
        for i, item in enumerate(inputs):
            sql = item.get(input_sql_key)
            question = item.get(input_question_key)
            evidence = item.get(input_evidence_key, '')
            db_id = item.get(input_db_id_key)

            if not question or str(question).strip() == '':
                continue

            if evidence:
                question = f'{question}\n{evidence}'

            try:
                create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
                db_details = '\n\n'.join([str(s) for s in create_statements])
                prompt = self._build_prompt(str(question), str(sql), db_details)
                prompts.append(prompt)
                valid_indices.append(i)
            except Exception as e:
                LOG.warning(f'Failed to build prompt for index {i}: {e}')
        return prompts, valid_indices

    def _generate_responses(self, prompts):
        responses = []
        for p in tqdm(prompts, desc='Checking correspondence'):
            try:
                responses.append(self.model(p))
            except Exception as e:
                LOG.error(f'Failed to check correspondence: {e}')
                responses.append('')
        return responses

    def _filter_results(self, inputs, valid_indices, responses):
        final_results = []
        for idx, response in enumerate(responses):
            if self._parse_consistency_response(response):
                final_results.append(inputs[valid_indices[idx]])

        LOG.info(f'Correspondence check results: {len(final_results)} passed, total {len(inputs)}')
        return final_results


class Text2SQLPromptGenerator(Text2SQLOps):
    def __init__(self, database_manager=None, prompt_template=None, **kwargs):
        super().__init__(**kwargs)
        self.database_manager = database_manager
        self.prompt_template = prompt_template

    def get_create_statements_and_insert_statements(self, db_id):
        return self.database_manager.get_create_statements_and_insert_statements(db_id)

    def _build_prompt(self, db_details, question, evidence, db_engine):
        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            return str(template.build_prompt(
                db_details=db_details,
                question=question,
                evidence=evidence,
                db_engine=db_engine
            ))

        return (
            f'Database Schema:\n{db_details}\n\n'
            f'Question: {question}\n'
            f'Evidence: {evidence}\n'
            f'Generate a SQL query for {db_engine}.'
        )

    def forward_batch_input(self, inputs, input_question_key='question', input_db_id_key='db_id',
                            input_evidence_key='evidence', output_prompt_key='prompt', **kwargs):
        if not inputs:
            return []
        if self.database_manager is None:
            raise ValueError('database_manager is required')

        db_engine = getattr(self.database_manager, 'db_type', 'unknown')
        final_results = []

        for item in tqdm(inputs, desc='Generating prompts'):
            db_id = item.get(input_db_id_key)
            question = item.get(input_question_key)
            evidence = item.get(input_evidence_key, '')

            if not db_id or not question:
                LOG.warning(f'Missing db_id or question for item: {item}')
                item[output_prompt_key] = ''
                final_results.append(item)
                continue

            db_id = str(db_id).replace('\n', '').replace('\r', '').strip()

            try:
                if hasattr(self.database_manager, 'get_db_details'):
                    db_details = self.database_manager.get_db_details(db_id)
                else:
                    create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
                    db_details = '\n\n'.join([str(s) for s in create_statements])

                prompt = self._build_prompt(
                    db_details=db_details,
                    question=question,
                    evidence=evidence,
                    db_engine=db_engine
                )
                item[output_prompt_key] = prompt
            except Exception as e:
                LOG.error(f'Failed to generate prompt for db_id={db_id}: {e}')
                item[output_prompt_key] = ''

            final_results.append(item)

        return final_results


class Text2SQLCoTGenerator(Text2SQLOps):
    def __init__(self, model=None, database_manager=None, prompt_template=None,
                 sampling_num=3, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.database_manager = database_manager
        self.prompt_template = prompt_template
        self.sampling_num = int(sampling_num)
        if self.sampling_num < 1:
            raise ValueError('sampling_num must be >= 1')
        sys_prompt = 'You are a database expert. Please generate a step-by-step reasoning ' \
                     '(Chain of Thought) and the final SQL.'
        self.model = model.share().prompt(sys_prompt) if model else None

    @staticmethod
    def get_desc(lang):
        if lang == 'zh':
            return (
                '对于每个条目，生成从自然语言问题和数据库Schema到SQL的CoT长链路推理过程。生成sampling_num条推理轨迹，但是不做验证'
                '输入参数：\n'
                '- input_sql_key: 输入SQL列名\n'
                '- input_question_key: 输入问题列名\n'
                '- input_db_id_key: 输入数据库ID列名\n\n'
                '输出参数：\n'
                '- output_cot_key: 输出CoT列名'
            )
        elif lang == 'en':
            return (
                'For each item, generate a CoT long chain of reasoning from natural language question '
                'and database Schema to SQL.\n'
                'Generate sampling_num reasoning trajectories, but do not verify.\n\n'
                'Input parameters:\n'
                '- input_sql_key: The name of the input SQL column\n'
                '- input_question_key: The name of the input question column\n'
                '- input_db_id_key: The name of the input database ID column\n\n'
                'Output parameters:\n'
                '- output_cot_key: The name of the output CoT column'
            )
        return 'CoT generator for Text2SQL tasks.'

    def _build_prompt(self, item, schema_str):
        question = item.get(self.input_question_key)
        gold_sql = item.get(self.input_sql_key)
        evidence = item.get(self.input_evidence_key, '')

        template = self.prompt_template
        if template is not None and hasattr(template, 'build_prompt'):
            return template.build_prompt(schema_str, question, gold_sql, evidence)

        return (
            f'Database Schema:\n{schema_str}\n\n'
            f'Question: {question}\n'
            f'Evidence: {evidence}\n'
            f'Target SQL: {gold_sql}\n\n'
            f'Please provide a detailed step-by-step reasoning that leads to the correct SQL query.'
        )

    def forward_batch_input(self, inputs, input_sql_key='SQL', input_question_key='question',
                            input_db_id_key='db_id', input_evidence_key='evidence',
                            output_cot_key='cot_responses', **kwargs):
        if not inputs:
            return []
        self._validate_manager()

        self.input_question_key = input_question_key
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.input_evidence_key = input_evidence_key

        prompts, mapping = self._prepare_cot_prompts(inputs, input_db_id_key)

        if not prompts:
            LOG.warning('No valid prompts generated for CoT')
            return inputs

        responses = self._generate_cot_responses(prompts)
        return self._group_results(inputs, mapping, responses, output_cot_key)

    def _validate_manager(self):
        if self.model is None:
            raise ValueError('model is required')
        if self.database_manager is None:
            raise ValueError('database_manager is required')

    def _prepare_cot_prompts(self, inputs, input_db_id_key):
        prompts = []
        mapping = []
        items_with_index = list(enumerate(inputs))

        for orig_idx, item in tqdm(items_with_index, desc='Preparing CoT prompts'):
            db_id = item.get(input_db_id_key)
            if not db_id:
                LOG.warning(f'Missing db_id for item index {orig_idx}')
                continue
            try:
                create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
                schema_str = '\n\n'.join([str(s) for s in create_statements])
                prompt = self._build_prompt(item, schema_str)
                for _ in range(self.sampling_num):
                    prompts.append(prompt)
                    mapping.append((orig_idx, item))
            except Exception as e:
                LOG.error(f'Failed to build prompt for item {orig_idx}: {e}')
        return prompts, mapping

    def _generate_cot_responses(self, prompts):
        responses = []
        for p in tqdm(prompts, desc='Generating CoT'):
            try:
                responses.append(self.model(p))
            except Exception as e:
                LOG.error(f'Failed to generate CoT: {e}')
                responses.append('')
        return responses

    def _group_results(self, inputs, mapping, responses, output_cot_key):
        grouped = defaultdict(list)
        for (orig_idx, _), resp in zip(mapping, responses):
            grouped[orig_idx].append(resp)

        results = []
        for orig_idx, item in enumerate(inputs):
            new_item = dict(item)
            new_item[output_cot_key] = grouped.get(orig_idx, [])
            results.append(new_item)
        return results


class Text2SQLCoTVotingGenerator(Text2SQLOps):
    def __init__(self, database_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.database_manager = database_manager
        self.tie_breaker = 'shortest_sql'

    def forward_batch_input(self, inputs, input_cot_responses_key='cot_responses',
                            input_db_id_key='db_id', output_cot_key='cot_reasoning', **kwargs):
        if not inputs:
            return []
        if self.database_manager is None:
            raise ValueError('database_manager is required')

        queries, mapping = self._prepare_queries(inputs, input_cot_responses_key, input_db_id_key)
        query_results = self._execute_queries(queries)
        item_candidates = self._collect_candidates(mapping, query_results)
        return self._vote_and_finalize(inputs, item_candidates, input_cot_responses_key, output_cot_key)

    def _prepare_queries(self, inputs, input_cot_responses_key, input_db_id_key):
        queries = []
        mapping = []

        for item_idx, item in enumerate(inputs):
            cot_responses = item.get(input_cot_responses_key, [])
            if not isinstance(cot_responses, list):
                continue
            db_id = item.get(input_db_id_key)
            if not db_id:
                continue
            for resp_idx, resp in enumerate(cot_responses):
                sql = _parse_sql_response(resp)
                if sql:
                    queries.append((str(db_id).strip(), sql))
                    mapping.append((item_idx, resp_idx, sql, resp))
        return queries, mapping

    def _execute_queries(self, queries):
        if not queries:
            return []
        try:
            return self.database_manager.batch_execute_queries(queries)
        except Exception as e:
            LOG.error(f'Failed to batch execute queries: {e}')
            return []

    def _collect_candidates(self, mapping, query_results):
        item_candidates = defaultdict(list)

        if len(query_results) != len(mapping):
            LOG.warning('Mismatch between queries and results or execution failed. Voting skipped for failed queries.')
            return item_candidates

        for (item_idx, _resp_idx, sql, cot), result in zip(mapping, query_results):
            signature = _result_to_signature(result)
            is_valid = result.success if hasattr(result, 'success') else False
            item_candidates[item_idx].append({
                'cot': cot,
                'sql': sql,
                'signature': signature,
                'is_valid': is_valid,
            })
        return item_candidates

    def _vote_and_finalize(self, inputs, item_candidates, input_cot_responses_key, output_cot_key):
        output_items = []
        drop_cot_responses = True

        for item_idx, item in enumerate(inputs):
            candidates = item_candidates.get(item_idx, [])
            chosen = _vote_select(candidates, self.tie_breaker)

            if chosen is None:
                cot_responses = item.get(input_cot_responses_key, [])
                chosen_cot = cot_responses[0] if cot_responses else ''
            else:
                chosen_cot = chosen['cot']

            if drop_cot_responses:
                out_item = {k: v for k, v in item.items() if k != input_cot_responses_key}
            else:
                out_item = dict(item)

            out_item[output_cot_key] = chosen_cot
            output_items.append(out_item)

        return output_items


class Schema:
    def __init__(self, schema):
        normalized = {
            table.strip().lower(): sorted({col.strip().lower() for col in cols})
            for table, cols in (schema or {}).items()
        }
        self._schema = normalized
        self._idMap = {
            '*': '__all__',
            **{f'{table}.{col}': f'__{table}.{col}__' for table, cols in normalized.items() for col in cols},
            **{table: f'__{table}__' for table in normalized},
        }

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

class SQLComponentClassifier(Text2SQLOps):
    Schema = Schema

    class EvalHardness:
        CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
        JOIN_KEYWORDS = ('join', 'on', 'as')

        WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
        UNIT_OPS = ('none', '-', '+', '*', '/')
        AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
        TABLE_TYPE = {
            'sql': 'sql',
            'table_unit': 'table_unit',
        }

        COND_OPS = ('and', 'or')
        SQL_OPS = ('intersect', 'union', 'except')
        ORDER_OPS = ('desc', 'asc')
        FUNC_OPS = ('cast', 'substring', 'date', 'round', 'coalesce')

        def __init__(self, schema, query):
            self.schema = schema
            self.query = query

        @property
        def tokenize(self):
            string = str(self.query)
            vals = {}

            def replace_literal(match):
                val = match.group(0)
                quote = val[0]
                content = val[1:-1]

                if quote == '`':
                    key = f'__col_{len(vals)}__'
                else:  # ' 或 "
                    key = f'__str_{len(vals)}__'

                vals[key] = content
                return key

            string = re.sub(r"(['\"`])(?:\\.|[^\\])*?\1", replace_literal, string)

            toks = [word.lower() for word in nltk.word_tokenize(string)]
            for i in range(len(toks)):
                if toks[i] in vals:
                    toks[i] = vals[toks[i]]

            eq_idxs = [idx for idx, tok in enumerate(toks) if tok == '=']
            eq_idxs.reverse()
            prefix = ('!', '>', '<')
            for eq_idx in eq_idxs:
                pre_tok = toks[eq_idx - 1]
                if pre_tok in prefix:
                    toks = toks[:eq_idx - 1] + [pre_tok + '='] + toks[eq_idx + 1:]

            return toks

        def scan_alias(self, toks):
            as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
            alias = {}
            for idx in as_idxs:
                alias[toks[idx + 1]] = toks[idx - 1]
            return alias

        def get_tables_with_alias(self, toks):
            tables = self.scan_alias(toks)
            for key in self.schema.schema:
                assert key not in tables, 'Alias {} has the same name in table'.format(key)
                tables[key] = key
            return tables

        def parse_col(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
            tok = toks[start_idx].strip().lower()
            if tok == '*':
                return start_idx + 1, schema.idMap[tok]

            if '.' in tok:
                alias, col = tok.split('.')
                key = tables_with_alias[alias] + '.' + col
                return start_idx + 1, schema.idMap[key]

            assert default_tables is not None and len(default_tables) > 0, 'Default tables should not be None or empty'

            for alias in default_tables:
                table = tables_with_alias[alias]
                if tok in schema.schema[table]:
                    key = table + '.' + tok
                    return start_idx + 1, schema.idMap[key]

            raise AssertionError('Error col: {}'.format(tok))

        def parse_col_unit(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
            idx = start_idx
            if toks[idx] in self.FUNC_OPS:
                func_name = toks[idx]
                idx += 1
                assert toks[idx] == '('
                idx += 1

                if func_name == 'cast':
                    idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                    assert toks[idx] == 'as'
                    idx += 1
                    data_type = toks[idx]
                    idx += 1
                    assert toks[idx] == ')'
                    idx += 1

                    func_call = ('func', func_name, col_id, data_type)
                    return idx, (self.AGG_OPS.index('none'), func_call, False)

                else:
                    idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                    while toks[idx] != ')':
                        idx += 1
                    idx += 1

                    func_call = ('func', func_name, col_id, None)
                    return idx, (self.AGG_OPS.index('none'), func_call, False)
            len_ = len(toks)
            isBlock = False
            isDistinct = False
            if toks[idx] == '(':
                isBlock = True
                idx += 1

            if toks[idx] in self.AGG_OPS:
                agg_id = self.AGG_OPS.index(toks[idx])
                idx += 1
                assert idx < len_ and toks[idx] == '('
                idx += 1
                if toks[idx] == 'distinct':
                    idx += 1
                    isDistinct = True
                idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                assert idx < len_ and toks[idx] == ')'
                idx += 1
                return idx, (agg_id, col_id, isDistinct)

            if toks[idx] == 'distinct':
                idx += 1
                isDistinct = True
            agg_id = self.AGG_OPS.index('none')
            idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)

            if isBlock:
                assert toks[idx] == ')'
                idx += 1

            return idx, (agg_id, col_id, isDistinct)

        def parse_val_unit(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
            idx = start_idx
            if toks[idx] in self.FUNC_OPS:
                func_name = toks[idx]
                idx += 1
                assert toks[idx] == '('
                idx += 1

                if func_name == 'cast':
                    idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                    assert toks[idx] == 'as'
                    idx += 1
                    data_type = toks[idx]
                    idx += 1
                    assert toks[idx] == ')'
                    idx += 1

                    func_call = ('func', func_name, col_id, data_type)
                    return idx, (self.AGG_OPS.index('none'), func_call, False)

                else:
                    idx, col_id = self.parse_col(toks, idx, tables_with_alias, schema, default_tables)
                    while toks[idx] != ')':
                        idx += 1
                    idx += 1

                    func_call = ('func', func_name, col_id, None)
                    return idx, (self.AGG_OPS.index('none'), func_call, False)
            len_ = len(toks)
            isBlock = False
            if toks[idx] == '(':
                isBlock = True
                idx += 1

            col_unit1 = None
            col_unit2 = None
            unit_op = self.UNIT_OPS.index('none')

            idx, col_unit1 = self.parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
            if idx < len_ and toks[idx] in self.UNIT_OPS:
                unit_op = self.UNIT_OPS.index(toks[idx])
                idx += 1
                idx, col_unit2 = self.parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

            if isBlock:
                assert toks[idx] == ')'
                idx += 1

            return idx, (unit_op, col_unit1, col_unit2)

        def parse_table_unit(self, toks, start_idx, tables_with_alias, schema):
            idx = start_idx
            len_ = len(toks)
            key = tables_with_alias[toks[idx]]

            if idx + 1 < len_ and toks[idx + 1] == 'as':
                idx += 3
            else:
                idx += 1

            return idx, schema.idMap[key], key

        def parse_value(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
            idx = start_idx
            len_ = len(toks)

            isBlock = False
            if toks[idx] == '(':
                isBlock = True
                idx += 1

            if toks[idx] == 'select':
                idx, val = self.parse_sql(toks, idx, tables_with_alias, schema)
            elif isinstance(toks[idx], str) and toks[idx] not in schema.idMap:
                val = toks[idx]
                idx += 1
            else:
                try:
                    val = float(toks[idx])
                    idx += 1
                except Exception:
                    end_idx = idx
                    while end_idx < len_ and toks[end_idx] not in (
                        ',', ')', 'and', *self.CLAUSE_KEYWORDS, *self.JOIN_KEYWORDS
                    ):
                        end_idx += 1

                    idx, val = self.parse_col_unit(toks[start_idx:end_idx], 0, tables_with_alias, schema, default_tables)
                    idx = end_idx

            if isBlock:
                assert toks[idx] == ')'
                idx += 1

            return idx, val

        def parse_condition(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
            idx = start_idx
            len_ = len(toks)
            conds = []

            while idx < len_:
                idx, val_unit = self.parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
                not_op = False
                if toks[idx] == 'not':
                    not_op = True
                    idx += 1

                assert idx < len_ and toks[idx] in self.WHERE_OPS, \
                    'Error condition: idx: {}, tok: {}'.format(idx, toks[idx])
                op_id = self.WHERE_OPS.index(toks[idx])
                idx += 1
                val1 = val2 = None
                if op_id == self.WHERE_OPS.index('between'):
                    idx, val1 = self.parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    assert toks[idx] == 'and'
                    idx += 1
                    idx, val2 = self.parse_value(toks, idx, tables_with_alias, schema, default_tables)
                else:
                    idx, val1 = self.parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    val2 = None

                conds.append((not_op, op_id, val_unit, val1, val2))

                if idx < len_ and (
                    toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (')', ';') or toks[idx] in self.JOIN_KEYWORDS
                ):
                    break

                if idx < len_ and toks[idx] in self.COND_OPS:
                    conds.append(toks[idx])
                    idx += 1

            return idx, conds

        def parse_select(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
            idx = start_idx
            len_ = len(toks)

            assert toks[idx] == 'select', "'select' not found"
            idx += 1
            isDistinct = False
            if idx < len_ and toks[idx] == 'distinct':
                idx += 1
                isDistinct = True
            val_units = []

            while idx < len_ and toks[idx] not in self.CLAUSE_KEYWORDS:
                agg_id = self.AGG_OPS.index('none')
                if toks[idx] in self.AGG_OPS:
                    agg_id = self.AGG_OPS.index(toks[idx])
                    idx += 1
                idx, val_unit = self.parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
                val_units.append((agg_id, val_unit))
                if idx < len_ and toks[idx] == ',':
                    idx += 1

            return idx, (isDistinct, val_units)

        def parse_from(self, toks, start_idx, tables_with_alias, schema):
            assert 'from' in toks[start_idx:], "'from' not found"

            len_ = len(toks)
            idx = toks.index('from', start_idx) + 1
            default_tables = []
            table_units = []
            conds = []

            while idx < len_:
                isBlock = False
                if toks[idx] == '(':
                    isBlock = True
                    idx += 1

                if toks[idx] == 'select':
                    idx, sql = self.parse_sql(toks, idx, tables_with_alias, schema)
                    table_units.append((self.TABLE_TYPE['sql'], sql))
                else:
                    if idx < len_ and toks[idx] == 'join':
                        idx += 1
                    idx, table_unit, table_name = self.parse_table_unit(toks, idx, tables_with_alias, schema)
                    table_units.append((self.TABLE_TYPE['table_unit'], table_unit))
                    default_tables.append(table_name)
                if idx < len_ and toks[idx] == 'on':
                    idx += 1
                    idx, this_conds = self.parse_condition(toks, idx, tables_with_alias, schema, default_tables)
                    if len(conds) > 0:
                        conds.append('and')
                    conds.extend(this_conds)

                if isBlock:
                    assert toks[idx] == ')'
                    idx += 1
                if idx < len_ and (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (')', ';')):
                    break

            return idx, table_units, conds, default_tables

        def parse_where(self, toks, start_idx, tables_with_alias, schema, default_tables):
            idx = start_idx
            len_ = len(toks)

            if idx >= len_ or toks[idx] != 'where':
                return idx, []

            idx += 1
            idx, conds = self.parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            return idx, conds

        def parse_group_by(self, toks, start_idx, tables_with_alias, schema, default_tables):
            idx = start_idx
            len_ = len(toks)
            col_units = []

            if idx >= len_ or toks[idx] != 'group':
                return idx, col_units

            idx += 1
            assert toks[idx] == 'by'
            idx += 1

            while idx < len_ and not (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (')', ';')):
                idx, col_unit = self.parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
                col_units.append(col_unit)
                if idx < len_ and toks[idx] == ',':
                    idx += 1
                else:
                    break

            return idx, col_units

        def parse_order_by(self, toks, start_idx, tables_with_alias, schema, default_tables):
            idx = start_idx
            len_ = len(toks)
            val_units = []
            order_type = 'asc'

            if idx >= len_ or toks[idx] != 'order':
                return idx, val_units

            idx += 1
            assert toks[idx] == 'by'
            idx += 1

            while idx < len_ and not (toks[idx] in self.CLAUSE_KEYWORDS or toks[idx] in (')', ';')):
                idx, val_unit = self.parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
                val_units.append(val_unit)
                if idx < len_ and toks[idx] in self.ORDER_OPS:
                    order_type = toks[idx]
                    idx += 1
                if idx < len_ and toks[idx] == ',':
                    idx += 1
                else:
                    break

            return idx, (order_type, val_units)

        def parse_having(self, toks, start_idx, tables_with_alias, schema, default_tables):
            idx = start_idx
            len_ = len(toks)

            if idx >= len_ or toks[idx] != 'having':
                return idx, []

            idx += 1
            idx, conds = self.parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            return idx, conds

        def parse_limit(self, toks, start_idx):
            idx = start_idx
            len_ = len(toks)

            if idx < len_ and toks[idx] == 'limit':
                idx += 2
                if not isinstance(toks[idx - 1], int):
                    return idx, 1

                return idx, int(toks[idx - 1])

            return idx, None

        def skip_semicolon(self, toks, start_idx):
            idx = start_idx
            while idx < len(toks) and toks[idx] == ';':
                idx += 1
            return idx

        def parse_sql(self, start_idx):
            toks = self.tokenize
            tables_with_alias = self.get_tables_with_alias(toks)
            schema = self.schema
            isBlock = False  # indicate whether this is a block of sql/sub-sql
            len_ = len(toks)
            idx = start_idx

            sql = {}
            if toks[idx] == '(':
                isBlock = True
                idx += 1

            from_end_idx, table_units, conds, default_tables = self.parse_from(
                toks, start_idx, tables_with_alias, schema
            )
            sql['from'] = {'table_units': table_units, 'conds': conds}
            # select clause
            _, select_col_units = self.parse_select(toks, idx, tables_with_alias, schema, default_tables)
            idx = from_end_idx
            sql['select'] = select_col_units
            # where clause
            idx, where_conds = self.parse_where(toks, idx, tables_with_alias, schema, default_tables)
            sql['where'] = where_conds
            # group by clause
            idx, group_col_units = self.parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
            sql['groupBy'] = group_col_units
            # having clause
            idx, having_conds = self.parse_having(toks, idx, tables_with_alias, schema, default_tables)
            sql['having'] = having_conds
            # order by clause
            idx, order_col_units = self.parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
            sql['orderBy'] = order_col_units
            # limit clause
            idx, limit_val = self.parse_limit(toks, idx)
            sql['limit'] = limit_val

            idx = self.skip_semicolon(toks, idx)
            if isBlock:
                assert toks[idx] == ')'
                idx += 1
            idx = self.skip_semicolon(toks, idx)

            # intersect/union/except clause
            for op in self.SQL_OPS:  # initialize IUE
                sql[op] = None
            if idx < len_ and toks[idx] in self.SQL_OPS:
                sql_op = toks[idx]
                idx += 1
                idx, IUE_sql = self.parse_sql(idx)
                sql[sql_op] = IUE_sql
            return idx, sql

        def has_agg(self, unit):
            return unit[0] != self.AGG_OPS.index('none')

        def count_agg(self, units):
            return len([unit for unit in units if self.has_agg(unit)])

        def get_nestedSQL(self, sql):
            nested = []
            for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
                if isinstance(cond_unit[3], dict):
                    nested.append(cond_unit[3])
                if isinstance(cond_unit[4], dict):
                    nested.append(cond_unit[4])
            if sql['intersect'] is not None:
                nested.append(sql['intersect'])
            if sql['except'] is not None:
                nested.append(sql['except'])
            if sql['union'] is not None:
                nested.append(sql['union'])
            return nested

        def count_component1(self, sql):
            count = 0
            if len(sql['where']) > 0:
                count += 1
            if len(sql['groupBy']) > 0:
                count += 1
            if len(sql['orderBy']) > 0:
                count += 1
            if sql['limit'] is not None:
                count += 1
            if len(sql['from']['table_units']) > 0:
                count += len(sql['from']['table_units']) - 1

            ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
            count += len([token for token in ao if token == 'or'])
            cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
            count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == self.WHERE_OPS.index('like')])

            return count

        def count_component2(self, sql):
            nested = self.get_nestedSQL(sql)
            return len(nested)

        def count_others(self, sql):
            count = 0
            # number of aggregation
            agg_count = self.count_agg(sql['select'][1])
            agg_count += self.count_agg(sql['where'][::2])
            agg_count += self.count_agg(sql['groupBy'])
            if len(sql['orderBy']) > 0:
                agg_count += self.count_agg(
                    [unit[1] for unit in sql['orderBy'][1] if unit[1]]
                    + [unit[2] for unit in sql['orderBy'][1] if unit[2]]
                )
            agg_count += self.count_agg(sql['having'])
            if agg_count > 1:
                count += 1

            # number of select columns
            if len(sql['select'][1]) > 1:
                count += 1

            # number of where conditions
            if len(sql['where']) > 1:
                count += 1

            # number of group by clauses
            if len(sql['groupBy']) > 1:
                count += 1

            return count

        def eval_hardness(self, sql):
            count_comp1_ = self.count_component1(sql)
            count_comp2_ = self.count_component2(sql)
            count_others_ = self.count_others(sql)

            if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
                return 'easy'
            elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                    (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
                return 'medium'
            elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                    (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                    (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
                return 'hard'
            else:
                return 'extra'

        def run(self):
            _, sql = self.parse_sql(0)
            hardness = self.eval_hardness(sql)
            return hardness

    class EvalHardnessLite:
        SCORE_RULES = (
            lambda sql: 2 if re.search(r'\( *select', sql) else 0,
            lambda sql: sql.count(' join '),
            lambda sql: 1 if sql.count(',') > 0 and 'from' in sql else 0,
            lambda sql: 1 if sql.count(' and ') + sql.count(' or ') >= 2 else 0,
            lambda sql: 1 if any(kw in sql for kw in ['in', 'exists', 'like']) else 0,
            lambda sql: 1 if 'group by' in sql else 0,
            lambda sql: 1 if 'having' in sql else 0,
            lambda sql: 1 if any(func in sql for func in ['cast', 'round', 'substring', 'date', 'coalesce']) else 0,
            lambda sql: 1 if 'order by' in sql else 0,
            lambda sql: 1 if 'limit' in sql else 0,
            lambda sql: 2 if any(op in sql for op in ['union', 'intersect', 'except']) else 0,
        )

        def __init__(self, sql, difficulty_config):
            self.sql = sql.lower()
            self.difficulty_config = difficulty_config

        def classify_difficulty(self, score):
            thresholds = self.difficulty_config['thresholds']
            labels = self.difficulty_config['labels']

            for i, threshold in enumerate(thresholds):
                if score <= threshold:
                    return labels[i]
            return labels[-1]

        def run(self):
            sql = self.sql
            score = sum(rule(sql) for rule in self.SCORE_RULES)

            select_cols = re.findall(r'select\s+(distinct\s+)?(.+?)\s+from', sql, re.DOTALL)
            if select_cols:
                num_commas = select_cols[0][1].count(',')
                if num_commas >= 1:
                    score += 1

            difficulty = self.classify_difficulty(score)
            return difficulty

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
        evaluator = self.EvalHardness(self.Schema(schema), sql)
        return evaluator.run()

    def eval_hardness_lite(self, sql):
        evaluator = self.EvalHardnessLite(str(sql), self.difficulty_config)
        return evaluator.run()

    def report_statistics(self, inputs, output_difficulty_key):
        difficulties = [item.get(output_difficulty_key) for item in inputs]
        counts = pd.Series(difficulties).value_counts()
        LOG.info('SQL Difficulty Statistics')
        difficulty_counts = {d: counts.get(d, 0) for d in ['easy', 'medium', 'hard', 'extra']}
        LOG.info(' | '.join([f'{d.title()}: {v}' for d, v in difficulty_counts.items()]))

    def forward_batch_input(self, inputs, input_sql_key='SQL',
                            output_difficulty_key='sql_component_difficulty', **kwargs):
        if not inputs:
            return []

        for item in tqdm(inputs, desc='Processing'):
            sql = item.get(input_sql_key)
            if not sql:
                item[output_difficulty_key] = 'unknown'
                continue
            hardness = self.eval_hardness_lite(str(sql))
            item[output_difficulty_key] = hardness

        self.report_statistics(inputs, output_difficulty_key)
        return inputs

class SQLExecutionClassifier(Text2SQLOps):
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
        sql_mapping = {}
        comparison_idx = 0
        for i, (predicted_sqls, ground_truth, db_id, idx) in enumerate(
            zip(predicted_sqls_list, ground_truth_list, db_ids, idxs)
        ):
            for j, predicted_sql in enumerate(predicted_sqls):
                comparisons.append((db_id, predicted_sql, ground_truth))
                sql_mapping[comparison_idx] = {
                    'original_idx': i,
                    'sql_idx': j,
                    'idx': idx,
                    'sql': predicted_sql
                }
                comparison_idx += 1
        return comparisons, sql_mapping

    @staticmethod
    def execute_model_batch(predicted_sqls_list, ground_truth_list, database_manager, db_ids, idxs, meta_time_out):
        comparisons, sql_mapping = SQLExecutionClassifier._prepare_comparisons(
            predicted_sqls_list, ground_truth_list, db_ids, idxs
        )

        try:
            batch_results = database_manager.batch_compare_queries(comparisons)
        except Exception as e:
            LOG.error(f'Batch comparison failed: {e}')
            results = []
            for _, (predicted_sqls, _, _, idx) in enumerate(zip(predicted_sqls_list, ground_truth_list, db_ids, idxs)):
                result_data = [{'res': 0, 'sql': sql, 'error': 'batch_failed'} for sql in predicted_sqls]
                results.append({'idx': idx, 'cnt_true': -1, 'results': result_data})
            return results

        results = {}
        for i, (predicted_sqls, _, _, idx) in enumerate(zip(predicted_sqls_list, ground_truth_list, db_ids, idxs)):
            results[i] = {'idx': idx, 'cnt_true': 0, 'results': [None] * len(predicted_sqls)}

        for batch_idx, comp_res in enumerate(batch_results):
            original_idx = sql_mapping[batch_idx]['original_idx']
            sql_idx = sql_mapping[batch_idx]['sql_idx']
            sql_str = sql_mapping[batch_idx]['sql']

            if comp_res['result1_success'] and comp_res['result2_success']:
                res = 1 if comp_res['equal'] else 0
                results[original_idx]['results'][sql_idx] = {'res': res, 'sql': sql_str, 'error': None}
                if res == 1:
                    results[original_idx]['cnt_true'] += 1
            else:
                error_msg = ('Predicted SQL failed; ' if not comp_res['result1_success'] else '') + \
                            ('Ground truth SQL failed' if not comp_res['result2_success'] else '')
                results[original_idx]['results'][sql_idx] = {'res': 0, 'sql': sql_str, 'error': error_msg}

        return [results[i] for i in sorted(results.keys())]

    def run_sqls_parallel(self, datas, database_manager, num_cpus, meta_time_out, predicted_sqls_list=None):
        if predicted_sqls_list is None:
            output_predicted_sqls_key = '_temp_predicted_sqls'
            predicted_sqls_list = [data_pair.get(output_predicted_sqls_key) for data_pair in datas]

        ground_truth_list = [data_pair.get(self.input_sql_key) for data_pair in datas]
        db_ids = [
            re.sub(r'[^A-Za-z0-9_]', '', str(data_pair.get(self.input_db_id_key)).replace('\n', '').strip())
            for data_pair in datas
        ]
        idxs = list(range(len(datas)))

        return SQLExecutionClassifier.execute_model_batch(
            predicted_sqls_list, ground_truth_list, database_manager, db_ids, idxs, meta_time_out
        )

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

    def forward_batch_input(self, inputs, input_db_id_key='db_id', input_sql_key='SQL',
                            input_prompt_key='prompt', output_difficulty_key='sql_execution_difficulty',
                            **kwargs):
        if not inputs:
            return []
        if self.model is None or self.database_manager is None:
            raise ValueError('model and database_manager are required')

        self.input_db_id_key = input_db_id_key
        self.input_sql_key = input_sql_key
        self.input_prompt_key = input_prompt_key

        input_prompts = [item.get(input_prompt_key) for item in inputs]
        LOG.info(f'Processing {len(input_prompts)} questions, generating {self.num_generations} SQLs each...')

        all_parsed_sqls = self._generate_and_parse_sqls(input_prompts)
        predicted_sqls_list = [
            all_parsed_sqls[i * self.num_generations: (i + 1) * self.num_generations]
            for i in range(len(inputs))
        ]

        exec_result = self.run_sqls_parallel(
            inputs, self.database_manager,
            num_cpus=os.cpu_count() or 1,
            meta_time_out=self.timeout,
            predicted_sqls_list=predicted_sqls_list
        )

        for execres in sorted(exec_result, key=lambda x: x['idx']) if exec_result else []:
            if execres is not None:
                inputs[execres['idx']][output_difficulty_key] = self.classify_difficulty(execres['cnt_true'])

        self.report_statistics(inputs, output_difficulty_key)
        return inputs
