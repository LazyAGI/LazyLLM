import json
import copy
import numpy as np

import lazyllm
from lazyllm.components.formatter import JsonFormatter
from .eval_base import BaseEvaluator


class ResponseRelevancy(BaseEvaluator):
    _default_generate_prompt_en = (
        'Please generate the most likely question based on '
        'the input, keeping it concise and to the point.')
    _default_generate_prompt_zh = ('请根据输入生成最可能的一个问题，保持简洁明了。')

    def __init__(self, llm, embedding, prompt=None, prompt_lang='en',
                 num_infer_questions=3, retry=3, concurrency=1):
        super().__init__(concurrency, retry)
        if prompt_lang.strip().lower() == 'zh':
            default_prompt = self._default_generate_prompt_zh
        else:
            default_prompt = self._default_generate_prompt_en
        self._llm = llm.prompt(prompt or default_prompt)
        self._embedding = embedding
        self._num_infer_questions = num_infer_questions
        self._necessary_keys = ['question', 'answer']

    def _cosine(self, x, y):
        product = np.dot(x, y)
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        raw_cosine = product / norm if norm != 0 else 0.0
        return max(0.0, min(raw_cosine, 1.0))

    def _process_one_data_impl(self, data):
        one_total_score = 0
        res = copy.deepcopy(data)
        res['infer_questions'] = []
        for _ in range(self._num_infer_questions):
            # Generate Questions:
            guess_question = self._execute_with_retries(data['answer'], self._llm)

            # Calculate Similarity:
            try:
                if isinstance(self._embedding, lazyllm.module.OnlineEmbeddingModuleBase):
                    vector1 = self._embedding(guess_question)
                    vector2 = self._embedding(data['question'])
                else:
                    vector1, vector2 = json.loads(self._embedding([guess_question, data['question']]))
                score = self._cosine(vector1, vector2)
            except Exception as e:
                lazyllm.LOG.error(f'Eval-Infer Error: {e}')
                score = 0
            res['infer_questions'].append({
                'question': guess_question,
                'score': round(score, 4)
            })
            one_total_score += score
        res['final_score'] = round(one_total_score / self._num_infer_questions, 4)
        return res


class Faithfulness(BaseEvaluator):
    _default_generate_prompt_en = (
        '[Task Description]\n'
        'Split the answer into independent factual statements using "|||" as '
        'the exclusive separator, following these rules:\n'
        '1. Each statement must be a complete sentence ending with proper punctuation\n'
        '2. Never use line breaks or other symbols as separators\n'
        '3. Statements containing "|||" must be rephrased\n'
        '4. Each statement must be clear, pronoun-free.\n'
        '[Output Format]\n'
        'statement_1|||statement_2|||statement_3\n'
        '[Example Input]\n'
        'Q: How does photosynthesis work?\n'
        'A: The process requires sunlight, then chlorophyll absorbs light energy. '
        'It converts water and CO2 into glucose.\n'
        '[Example Output]\n'
        'Photosynthesis requires sunlight.|||Chlorophyll absorbs light energy.'
        '|||Chlorophyll converts water and CO2 into glucose.\n'
    )
    _default_eval_prompt_en = (
        '[Task Description]\n'
        'Evaluate each "|||"-separated statement against provided context using binary scoring:\n'
        'Fully supported by context: 1\n'
        'Unsupported/contradictory: 0\n'
        '[Output Requirements]\n'
        '1. JSON format with array of objects\n'
        '2. Each object contains:\n'
        '    - "statement": Original text\n'
        '    - "score": 1 or 0\n'
        '3. Wrap output in ```json code block\n'
        '[Example Input]\n'
        'Context: Photosynthesis occurs in chloroplasts. Light reactions produce ATP using sunlight. '
        'Calvin cycle fixes CO2 into sugars.\n'
        'Statements: Photosynthesis requires sunlight.|||Chlorophyll absorbs light energy.'
        '|||Chlorophyll converts water and CO2 into glucose.\n'
        '[Example Output]\n'
        '[{"statement": "Photosynthesis requires sunlight.","score": 1},'
        '{"statement": "Chlorophyll absorbs light energy.", "score": 1},'
        '{"statement": "Chlorophyll converts water and CO2 into glucose.","score": 0}]\n'
    )
    _default_generate_prompt_zh = (
        '[任务描述]\n'
        '使用"|||"作为唯一分隔符，将答案分割成独立的事实陈述，遵循以下规则：\n'
        '1. 每个陈述必须是完整的句子，并以适当的标点结束\n'
        '2. 不要使用换行符或其他符号作为分隔符\n'
        '3. 包含"|||"的陈述必须重新措辞\n'
        '4. 每个陈述必须清晰，不包含代词。\n'
        '[输出格式]\n'
        'statement_1|||statement_2|||statement_3\n'
        '[示例输入]\n'
        'Q: 光合作用是如何工作的？\n'
        'A: 该过程需要阳光，然后叶绿素吸收光能。它将水和CO2转化为葡萄糖。\n'
        '[示例输出]\n'
        '光合作用需要阳光。|||叶绿素吸收光能。|||叶绿素将水和CO2转化为葡萄糖。\n'
    )
    _default_eval_prompt_zh = (
        '[任务描述]\n'
        '使用二进制评分对每个"|||"分隔的陈述与提供的内容进行评估：\n'
        '完全由内容支持：1\n'
        '不支持/矛盾：0\n'
        '[输出要求]\n'
        '1. JSON格式，包含对象数组\n'
        '2. 每个对象包含：\n'
        '    - "statement": 原始文本\n'
        '    - "score": 1或0\n'
        '3. 将输出包裹在```json代码块中\n'
        '[示例输入]\n'
        'Context: 光合作用发生在叶绿体中。光反应利用阳光产生ATP。卡尔文循环将CO2固定成糖。\n'
        'Statements: 光合作用需要阳光。|||叶绿素吸收光能。|||叶绿素将水和CO2转化为葡萄糖。\n'
        '[示例输出]\n'
        '[{"statement": "光合作用需要阳光。","score": 1},'
        '{"statement": "叶绿素吸收光能。", "score": 1},'
        '{"statement": "叶绿素将水和CO2转化为葡萄糖。","score": 0}]\n'
    )

    def __init__(self, llm, generate_prompt=None, eval_prompt=None, prompt_lang='en', retry=3, concurrency=1):
        super().__init__(concurrency, retry)
        self._base_llm = llm
        if prompt_lang == 'zh':
            default_generate_prompt = generate_prompt or self._default_generate_prompt_zh
            default_eval_prompt = eval_prompt or self._default_eval_prompt_zh
        else:
            default_generate_prompt = generate_prompt or self._default_generate_prompt_en
            default_eval_prompt = eval_prompt or self._default_eval_prompt_en
        self._build_llms(self._base_llm, default_generate_prompt, default_eval_prompt)
        self._necessary_keys = ['question', 'answer', 'context']

    def _build_llms(self, base_llm, generate_prompt, eval_prompt):
        self._gene_llm = base_llm.share(prompt=generate_prompt)
        self._eval_llm = base_llm.share(prompt=eval_prompt).formatter(JsonFormatter())

    def _validate_eval_result(self, result):
        return (
            isinstance(result, list)
            and len(result) > 0
            and all(isinstance(i, dict) and 'score' in i for i in result)
        )

    def _post_processor(self, eval_result):
        if isinstance(eval_result, dict):
            eval_result = [eval_result]
        return eval_result

    def _process_one_data_impl(self, data):
        res = copy.deepcopy(data)
        # Generate Statements:
        query1 = f'Q: {data["question"]}\nA: {data["answer"]}'
        statements = self._execute_with_retries(query1, self._gene_llm)
        res['statements'] = statements

        # Eval Statements in Context:
        query2 = f'Context: {data["context"]}\nStatements: {statements}'
        eval_result = self._execute_with_retries(
            query2, self._eval_llm, self._validate_eval_result, self._post_processor)
        if not self._validate_eval_result(eval_result):
            lazyllm.LOG.error("Invalid evaluation result format")
            res.update({'scores': [], 'final_score': 0.0})
            return res

        total_score = sum(
            int(entry.get('score', 0)) if entry.get('score') in (0, 1) else 0
            for entry in eval_result
        )
        res['scores'] = eval_result
        res['final_score'] = round(total_score / len(eval_result), 4) if eval_result else 0.0
        return res
