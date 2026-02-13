from ..base_data import data_register
from lazyllm import TrainableModule
from lazyllm.components.formatter import JsonFormatter
from collections import Counter
from lazyllm.tools.data.operators.utils import boxed_res_extractor
from lazyllm.thirdparty import math_verify

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'

GenCot = data_register.new_group('genCot')

class CoTGenerator(GenCot):
    def __init__(self,
                 input_key='query',
                 output_key='cot_answer',
                 model=None,
                 user_prompt=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.user_prompt = user_prompt

        output_structure = f'''
        输出格式要求：
        {{
            "{self.output_key}": "包含CoT推理过程和最终boxed答案"
        }}
        '''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data):
        question = data.get(self.input_key, '')
        if not question:
            data[self.output_key] = None
            return data

        base_prompt = f'''
        问题：
        {question}

        规则：
        - 输出详细CoT
        - 最终答案必须使用 \boxed{{ANSWER}} 包裹
        '''

        if self.user_prompt is None:
            user_prompt = '请为这个问题生成带有思维链（Chain-of-Thought, CoT）的输出结果：\n' + base_prompt
        else:
            user_prompt = self.user_prompt + '\n' + f'问题：{question}'

        res = self.model(user_prompt)
        data[self.output_key] = res.get(self.output_key, None)
        return data


class SelfConsistencyCoTGenerator(GenCot):
    def __init__(self,
                 input_key='query',
                 output_key='cot_answer',
                 num_samples=5,
                 model=None,
                 user_prompt=None,
                 **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.num_samples = num_samples
        self.user_prompt = user_prompt

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

    def _build_prompt(self, question):
        base_prompt = f'''
        问题：
        {question}

        规则：
        - 输出详细CoT
        - 最终答案必须使用 \boxed{{ANSWER}} 包裹
        '''
        if self.user_prompt is None:
            return '请为这个问题生成带有思维链（Chain-of-Thought, CoT）的输出结果：\n' + base_prompt
        return self.user_prompt + '\n' + f'问题：{question};'

    def forward(self, data):
        question = data.get(self.input_key, '')
        if not question:
            data[self.output_key] = None
            return data

        cot_list = []
        boxed_answers = []

        prompt = self._build_prompt(question)
        candidates = []
        for _ in range(self.num_samples):
            response = self.model(prompt)
            cot = response
            boxed = boxed_res_extractor(response)
            candidates.append(boxed)
            if boxed is not None:
                cot_list.append(cot)
                boxed_answers.append(boxed)

        if not boxed_answers:
            data[self.output_key] = None
            return data

        counter = Counter(boxed_answers)
        majority_answer = counter.most_common(1)[0][0]
        data['candidates'] = candidates
        for cot, ans in zip(cot_list, boxed_answers):
            if ans == majority_answer:
                data[self.output_key] = cot
                return data

        data[self.output_key] = None
        return data

@data_register('data.genCot', rewrite_func='forward')
def answer_verify(data, answer_key='reference', infer_key='llm_extracted', output_key='is_equal'):
    real_answer = data.get(answer_key, None)
    llm_answer = data.get(infer_key, None)

    if real_answer is None or llm_answer is None:
        data[output_key] = False
        return data

    try:
        parsed_real = math_verify.parse(str(real_answer))
        parsed_llm = math_verify.parse(str(llm_answer))
        data[output_key] = math_verify.verify(parsed_real, parsed_llm)

    except Exception:
        data[output_key] = False

    return data
