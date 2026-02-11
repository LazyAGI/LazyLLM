from ..base_data import data_register
import regex
from lazyllm import TrainableModule, LOG
import re
from lazyllm.thirdparty import transformers
from lazyllm.components.formatter import JsonFormatter

DEFAULT_MODEL = 'qwen2.5-0.5B-instruct'
DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B'
MathQA = data_register.new_group('mathQA')


def boxed_extractor(text):
    if not isinstance(text, str):
        return None
    pattern = r'\\boxed\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    matches = regex.findall(pattern, text)
    return matches[-1].strip() if matches else None


@data_register('data.mathQA', rewrite_func='forward')
def math_answer_extractor(data, input_key='answer', output_key='math_answer'):
    assert isinstance(data, dict)
    answer = data[input_key]
    math_answer = boxed_extractor(answer)
    data[output_key] = math_answer
    return data

class MathAnswerGenerator(MathQA):
    def __init__(self,
                 input_key='question',
                 output_key='answer',
                 regenerate_key='regenerate',
                 model=None,
                 user_prompt=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.regenerate_key = regenerate_key
        self.user_prompt = user_prompt

        output_structure = f'''
        输出格式要求：
        {{
            "{self.output_key}": "推理结果"
        }}
        '''

        self.model = model or TrainableModule(DEFAULT_MODEL)

        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data):

        answer = data.get(self.output_key)
        regenerate = data.get(self.regenerate_key, False)

        if answer is not None and regenerate is False:
            return None

        question = data.get(self.input_key)

        base_prompt = f'''
        问题：
        {question}

        规则：
        - 输出详细的过程
        - 最终结果使用 \\boxed{{ANSWER}} 包裹
        '''

        if self.user_prompt is None:
            prompt = '请为这个数学问题生成推理结果。\n' + base_prompt
        else:
            prompt = self.user_prompt + '\n' + f'问题：{question}'

        res = self.model(prompt)

        data[self.output_key] = res.get(self.output_key)
        data[self.regenerate_key] = False

        return data


class DifficultyEvaluator(MathQA):
    def __init__(self,
                 input_key='question',
                 output_key='difficulty',
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
            "{self.output_key}": "Easy | Medium | Hard"
        }}
        '''

        self.model = model or TrainableModule(DEFAULT_MODEL)

        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data):

        if data.get(self.output_key) is not None:
            return None

        question = data.get(self.input_key)

        base_prompt = f'''
        问题：
        {question}

        难度级别：
        - Easy : 小学
        - Medium : 初中/高中
        - Hard : 大学及以上

        规则：
        - 只能输出 Easy / Medium / Hard
        '''

        if self.user_prompt is None:
            prompt = '判断下面数学问题的难度。\n' + base_prompt
        else:
            prompt = self.user_prompt + '\n' + f'问题：{question}'

        res = self.model(prompt)

        data[self.output_key] = res.get(self.output_key)
        return data


@data_register(
    'data.mathQA',
    rewrite_func='forward_batch_input'
)
def DifficultyEvaluatorBatch(data, input_key='difficulty'):
    result = {}
    for entry in data:
        key = entry.get(input_key)
        if key in result:
            result[key] += 1
        else:
            result[key] = 1
    return [result]


class QualityEvaluator(MathQA):
    def __init__(self,
                 question_key='question',
                 answer_key='answer',
                 output_key='score',
                 model=None,
                 user_prompt=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.question_key = question_key
        self.answer_key = answer_key
        self.output_key = output_key
        self.user_prompt = user_prompt

        output_structure = f'''
        输出格式要求：
        {{
            "{self.output_key}": 0
        }}
        '''

        self.model = model or TrainableModule(DEFAULT_MODEL)

        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data):

        if data.get(self.output_key) is not None:
            return None

        question = data.get(self.question_key)
        answer = data.get(self.answer_key)

        base_prompt = f'''
        问题：
        {question}

        答案：
        {answer}

        规则：
        - 输出 0 表示需要重新生成
        - 输出 1 表示质量合格
        '''

        if self.user_prompt is None:
            prompt = '请检查问题和答案的质量。\n' + base_prompt
        else:
            prompt = self.user_prompt + '\n' + f'问题：{question}; 答案: {answer}'

        res = self.model(prompt)

        data[self.output_key] = res.get(self.output_key)
        return data


class DuplicateAnswerDetector(MathQA):
    def __init__(self,
                 question_key='question',
                 answer_key='answer',
                 output_key='duplicate',
                 min_repeat_len=15,
                 repeat_threshold=2,
                 periodic_min_repeat=3,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.question_key = question_key
        self.answer_key = answer_key
        self.output_key = output_key

        self.min_repeat_len = min_repeat_len
        self.repeat_threshold = repeat_threshold
        self.periodic_min_repeat = periodic_min_repeat

    def _is_periodic(self, text):
        n = len(text)
        if n < 6:
            return False
        for size in range(1, n // 2 + 1):
            if n % size != 0:
                continue

            unit = text[:size]
            if unit * (n // size) == text:
                if (n // size) >= self.periodic_min_repeat:
                    return True

        return False

    def _has_long_repeat(self, merged_text):
        seen = {}
        text_len = len(merged_text)

        for i in range(text_len - self.min_repeat_len + 1):

            substr = merged_text[i:i + self.min_repeat_len]

            if not substr.strip():
                continue

            seen[substr] = seen.get(substr, 0) + 1

            if seen[substr] >= self.repeat_threshold:
                return True

        return False

    def _sentence_repeat(self, answer):
        sentences = re.split(r'[。！？.!?\n]', answer)
        counter = {}
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            counter[s] = counter.get(s, 0) + 1
            if counter[s] >= 3:
                return True
        return False

    def forward(self, data):
        assert isinstance(data, dict)
        question = str(data.get(self.question_key, '') or '')
        answer = str(data.get(self.answer_key, '') or '')
        data[self.output_key] = False
        if not answer:
            return data

        merged = question + '\n' + answer
        if self._is_periodic(answer):
            data[self.output_key] = True
            return data

        if self._sentence_repeat(answer):
            data[self.output_key] = True
            return data

        if self._has_long_repeat(merged):
            data[self.output_key] = True
            return data

        return data

class ReasoningAnswerTokenLengthFilter(MathQA):
    def __init__(self,
                 input_key='answer',
                 max_answer_token_length=300,
                 tokenize=True,
                 tokenizer=None,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.max_answer_token_length = max_answer_token_length
        self.tokenizer = tokenizer

        if tokenize and tokenizer is None:
            LOG.warning(
                f'tokenize=True but tokenizer is None, '
                f'loading tokenizer from default model: {DEFAULT_TOKENIZER}'
            )
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    DEFAULT_TOKENIZER,
                    trust_remote_code=True
                )
                self.tokenize = True
            except Exception as e:
                LOG.warning(
                    f'failed to load tokenizer from {DEFAULT_TOKENIZER}, '
                    f'falling back to char count, error: {e}'
                )
                self.tokenize = False
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
            self.tokenize = tokenize

        self.empty_count = 0

    def _get_len(self, text: str):
        if text is None or (isinstance(text, str) and text.strip() == ''):
            self.empty_count += 1
            return self.max_answer_token_length + 1

        try:
            if self.tokenize:
                return len(
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False
                    )
                )
            return len(text)

        except Exception as e:
            LOG.warning(f'token encode failed: {e}')
            self.empty_count += 1
            return self.max_answer_token_length + 1

    def forward(self, data: dict):
        text = data.get(self.input_key, '')
        if not text:
            self.empty_count += 1
            return []

        token_len = self._get_len(text)

        if token_len <= self.max_answer_token_length:
            return None

        # clear eligible answer
        data[self.input_key] = ''
        return data

class QuestionFusionGenerator(MathQA):
    def __init__(self,
                 input_key='question',
                 output_key='answer',
                 model=None,
                 user_prompt=None,
                 list_key='question_list',
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)

        self.input_key = input_key
        self.output_key = output_key
        self.user_prompt = user_prompt
        self.list_key = list_key

        output_structure = f'''
        输出格式要求：
        {{
            "{input_key}": "融合后的问题",
            "{self.output_key}": "推理结果"
        }}
        '''

        self.model = model or TrainableModule(DEFAULT_MODEL)

        self.model.prompt(output_structure)\
            .formatter(JsonFormatter())\
            .start()

    def forward(self, data):
        questions = data.get(self.list_key, [])
        assert len(questions) > 1
        base_prompt = f'''
        问题列表：
        {questions}

        规则：
        - 融合列表中的问题，输出一个新问题
        - 输出详细的过程
        - 最终结果使用 \\boxed{{ANSWER}} 包裹
        '''

        if self.user_prompt is None:
            prompt = base_prompt
        else:
            prompt = self.user_prompt\
                 + '\n' + f'融合列表中的问题，输出一个新问题：{questions}'

        res = self.model(prompt)

        data[self.output_key] = res.get(self.output_key)

        return data
