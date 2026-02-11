from ..base_data import data_register
from lazyllm import TrainableModule
from lazyllm.components.formatter import JsonFormatter

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'

EnQA = data_register.new_group('enQA')


class QueryRewriter(EnQA):

    def __init__(self,
                 input_key='query',
                 output_key='rewrite_querys',
                 rewrite_num=3,
                 model=None,
                 user_prompt=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.rewrite_num = rewrite_num
        self.user_prompt = user_prompt

        output_structure = f'''
        输出格式要求：
        {{
            "{self.output_key}": ["rewrite1","rewrite2"]
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

        query = data.get(self.input_key)
        if not query:
            return None

        if data.get(self.output_key) is not None:
            return None

        base_prompt = f'''
        原问题：
        {query}

        规则：
        - 生成 {self.rewrite_num} 个不同表达
        - 保持语义一致
        - 不要解释
        '''

        if self.user_prompt is None:
            prompt = '请重写下面的问题，使其语义一致但表达不同。\n' + base_prompt
        else:
            prompt = self.user_prompt + \
                '\n' + f'原问题：{query} \n 生成 {self.rewrite_num} 个不同表达'

        res = self.model(prompt)

        data[self.output_key] = res.get(self.output_key, [])
        return data


class DiversityScorer(EnQA):

    def __init__(self,
                 input_key='rewrite_querys',
                 output_key='diversity_querys',
                 model=None,
                 user_prompt=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.user_prompt = user_prompt

        output_structure = '''
        输出格式要求：
        {
            "diversity_scores": [0,1]
        }
        '''

        if model is None:
            self.model = TrainableModule(DEFAULT_MODEL)
        else:
            self.model = model

        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data):
        querys = data.get(self.input_key)
        if not querys:
            return None

        if data.get(self.output_key) is not None:
            return None

        base_prompt = f'''
        问题列表：
        {querys}

        规则：
        - 表达重复或相似度高：score = 0
        - 表达差异明显：score = 1
        - 输出与输入顺序一致
        '''

        if self.user_prompt is None:
            prompt = '判断下面问题列表的表达多样性。\n' + base_prompt
        else:
            prompt = self.user_prompt + '\n' + f'问题列表：{querys};'

        res = self.model(prompt)

        scores = res.get('diversity_scores', [])

        new_list = []
        for i, q in enumerate(querys):
            score = scores[i] if i < len(scores) else 0
            new_list.append({
                'rewritten_query': q,
                'diversity_score': score
            })

        data[self.output_key] = new_list
        return data

@data_register('data.enQA', rewrite_func='forward')
def post_processor(data, input_key):
    items = data.get(input_key)
    if not items:
        return None

    result = []
    for obj in items:

        if not isinstance(obj, dict):
            continue

        new_row = data.copy()
        new_row.pop(input_key, None)
        for k, v in obj.items():
            new_row[k] = v

        result.append(new_row)

    return result

@data_register('data.enQA', rewrite_func='forward')
def diversity_filter(data, input_key, min_score):
        score = data.get(input_key, 0)
        if score >= min_score:
            return None
        return []
