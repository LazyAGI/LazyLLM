from ..base_data import data_register
from lazyllm import LOG
from lazyllm.components.formatter import JsonFormatter

PreferenceOps = data_register.new_group('preference_ops')

class IntentExtractor(PreferenceOps):
    def __init__(self, model=None, input_key='content', output_key='intent', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = '你是一个意图提取助手，请从用户文本中提取核心意图，并以 JSON 格式返回。'
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.input_key in data:
            data[self.output_key] = self.extract(data[self.input_key])
        return data

    def extract(self, raw_text):
        instruction = f'提炼以下用户文本的核心意图: \n{raw_text}'
        res = self.model(instruction)
        return res if isinstance(res, dict) else None


class PreferenceResponseGenerator(PreferenceOps):
    def __init__(self, model=None, n=3, temperature=1.0, system_prompt=None,
                 input_key='intent', output_key='responses', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.n = n
        self.temperature = temperature
        self.input_key = input_key
        self.output_key = output_key
        self.model = model.share()
        if system_prompt:
            self.model = self.model.prompt(system_prompt)

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.input_key in data:
            data[self.output_key] = self.generate(data[self.input_key])
        return data

    def generate(self, x):
        responses = []
        for _ in range(self.n):
            response = self.model(x, temperature=self.temperature)
            responses.append(response)
        return responses


class ResponseEvaluator(PreferenceOps):
    def __init__(self, model=None, input_key='content', response_key='responses', output_key='evaluation', **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.response_key = response_key
        self.output_key = output_key
        sys_prompt = (
            '你是一个专业的回复评测判官。请针对用户提供的指令和回复，从以下三个维度进行打分，总分为 10 分：\n'
            '1. 有用性 (Helpfulness): 满分 4 分。回复是否解决了用户的问题。\n'
            '2. 真实性 (Truthfulness): 满分 3 分。回复内容是否准确、无误导。\n'
            '3. 流畅度 (Fluency): 满分 3 分。回复是否自然、逻辑清晰。\n'
            '请先给出详细的理由 (Rationale)，然后以 JSON 格式输出各项得分及总分。\n'
            '输出示例：\n'
            '{\n'
            '  "rationale": "回复简洁且准确...",\n'
            '  "scores": {"helpfulness": 4, "truthfulness": 3, "fluency": 3},\n'
            '  "total_score": 10\n'
            '}'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.input_key in data and self.response_key in data:
            data[self.output_key] = self.evaluate(data[self.input_key], data[self.response_key])
        return data

    def evaluate(self, instruction, responses):
        scores = []
        for resp in responses:
            prompt = (
                f'指令: {instruction}\n\n'
                f'回复: {resp}\n\n'
                '请对上述回复进行打分。'
            )
            res = self.model(prompt)
            if isinstance(res, dict):
                scores.append(res.get('total_score', 0))
            else:
                LOG.warning(f'Failed to extract total_score from response: {res}')
                scores.append(0)
        return scores


class PreferencePairConstructor(PreferenceOps):
    def __init__(self, strategy='max_min', threshold=0.5,
                 instruction_key='intent', response_key='responses', score_key='evaluation',
                 output_chosen_key='chosen', output_rejected_key='rejected', **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self.threshold = threshold
        self.instruction_key = instruction_key
        self.response_key = response_key
        self.score_key = score_key
        self.output_chosen_key = output_chosen_key
        self.output_rejected_key = output_rejected_key

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        if self.response_key in data and self.score_key in data:
            responses = data[self.response_key]
            scores = data[self.score_key]

            if not responses or not scores or len(responses) != len(scores):
                return []

            chosen, rejected = self.construct_pair(responses, scores)

            if chosen is not None and rejected is not None:
                return {
                    'instruction': data.get(self.instruction_key, ''),
                    self.output_chosen_key: chosen,
                    self.output_rejected_key: rejected
                }

        return []

    def construct_pair(self, responses, scores):
        if len(responses) < 2:
            return None, None

        pairs = list(zip(responses, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)

        if self.strategy == 'max_min':
            chosen_pair = pairs[0]
            rejected_pair = pairs[-1]

            if chosen_pair[1] > rejected_pair[1]:
                return chosen_pair[0], rejected_pair[0]

        elif self.strategy == 'threshold':
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    score_diff = pairs[i][1] - pairs[j][1]
                    if score_diff >= self.threshold:
                        return pairs[i][0], pairs[j][0]

        return None, None
