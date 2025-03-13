import json
import copy
import numpy as np

import lazyllm
from lazyllm.components.formatter import JsonFormatter
from .eval_base import BaseEvaluator


class ResponseRelevancy(BaseEvaluator):
    default_generate_prompt_en = (
        'Please generate the most likely question based on '
        'the input, keeping it concise and to the point.')
    default_generate_prompt_zh = ('请根据输入生成最可能的一个问题，保持简洁明了。')

    def __init__(self, llm, embedding, prompt=None, prompt_lang='en',
                 num_infer_questions=3, retry=3, concurrency=1):
        super().__init__(concurrency, retry)
        if prompt_lang.strip().lower() == 'zh':
            default_prompt = self.default_generate_prompt_zh
        else:
            default_prompt = self.default_generate_prompt_en
        self.llm = llm.prompt(prompt or default_prompt)
        self.embedding = embedding
        self.num_infer_questions = num_infer_questions
        self.necessary_keys = ['question', 'answer']

    def cosine(self, x, y):
        product = np.dot(x, y)
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        raw_cosine = product / norm if norm != 0 else 0.0
        return max(0.0, min(raw_cosine, 1.0))

    def process_one_data(self, data):
        one_total_score = 0
        res = copy.deepcopy(data)
        res['infer_questions'] = []
        for _ in range(self.num_infer_questions):
            # Generate Questions:
            guess_question = self.llm_infer(data['answer'], self.llm)

            # Calculate Similarity:
            try:
                if isinstance(self.embedding, lazyllm.module.OnlineEmbeddingModuleBase):
                    vector1 = self.embedding(guess_question)
                    vector2 = self.embedding(data['question'])
                else:
                    vector1, vector2 = json.loads(self.embedding([guess_question, data['question']]))
                score = self.cosine(vector1, vector2)
            except Exception as e:
                lazyllm.LOG.error(f'Eval-Infer Error: {e}')
                score = 0
            res['infer_questions'].append({
                'question': guess_question,
                'score': score
            })
            one_total_score += score
        res['final_score'] = one_total_score / self.num_infer_questions
        return res


class Faithfulness(BaseEvaluator):
    default_generate_prompt_en = (
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
    default_eval_prompt_en = (
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
    default_generate_prompt_zh = (
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
    default_eval_prompt_zh = (
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
        self.base_llm = llm
        if prompt_lang == 'zh':
            default_generate_prompt = generate_prompt or self.default_generate_prompt_zh
            default_eval_prompt = eval_prompt or self.default_eval_prompt_zh
        else:
            default_generate_prompt = generate_prompt or self.default_generate_prompt_en
            default_eval_prompt = eval_prompt or self.default_eval_prompt_en
        self.build_llms(self.base_llm, default_generate_prompt, default_eval_prompt)
        self.necessary_keys = ['question', 'answer', 'context']

    def build_llms(self, base_llm, generate_prompt, eval_prompt):
        self.gene_llm = base_llm.share(prompt=generate_prompt)
        self.eval_llm = base_llm.share(prompt=eval_prompt).formatter(JsonFormatter())

    def process_one_data(self, data):
        res = copy.deepcopy(data)
        # Generate Statements:
        query1 = f'Q: {data["question"]}\nA: {data["answer"]}'
        statements = self.llm_infer(query1, self.gene_llm)
        res['statements'] = statements

        # Eval Statements in Context:
        query2 = f'Context: {data["context"]}\nStatements: {statements}'
        score_str_json = self.llm_infer(query2, self.eval_llm)
        res['scores'] = score_str_json

        # Caculate Score:
        one_total_score = 0
        for item in score_str_json:
            score = int(item.get('score', 0))
            one_total_score += score if score in (0, 1) else 0
        res['final_score'] = one_total_score / len(score_str_json)
        return res
