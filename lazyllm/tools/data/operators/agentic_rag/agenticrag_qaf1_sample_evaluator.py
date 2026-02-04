'''AgenticRAG QA F1 Sample Evaluator operator'''
import re
import string
from collections import Counter
from lazyllm import LOG
from ...base_data import data_register
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# 获取或创建 agenticrag 组（确保所有模块共享同一个组）
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


class AgenticRAGQAF1SampleEvaluator(agenticrag):
    '''
    Evaluator for computing F1 scores between predicted answers and reference answers.
    用于评估预测答案与多个参考答案之间的 F1 分数

    使用 forward 模式：每条数据独立计算 F1 分数，框架自动多进程并发。
    '''

    def __init__(self,
                 prediction_key: str = 'refined_answer',
                 ground_truth_key: str = 'golden_doc_answer',
                 output_key: str = 'F1Score',
                 _concurrency_mode: str = 'process',  # 纯计算，使用多进程
                 **kwargs):
        super().__init__(_concurrency_mode=_concurrency_mode, **kwargs)
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key
        self.output_key = output_key
        LOG.info(f'Initializing {self.__class__.__name__}...')

    @staticmethod
    def get_desc(lang: str = 'zh'):
        if lang == 'zh':
            return (
                '用于评估预测答案与多个参考答案之间的 F1 分数。\n\n'
                '输入参数：\n'
                '- prediction_key: 预测答案字段名（默认值：\'refined_answer\'）\n'
                '- ground_truth_key: 真实答案字段名（默认值：\'golden_doc_answer\'）\n'
                '- output_key: 输出F1分数字段名（默认值：\'F1Score\'）\n'
            )
        elif lang == 'en':
            return 'Evaluate F1 scores between predicted answers and reference answers.'
        else:
            return 'Evaluate F1 scores between predicted answers and reference answers.'

    def normalize_answer(self, s: str) -> str:
        '''Normalize answer text by removing articles, punctuation, and fixing whitespace'''
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(self, prediction: str, ground_truths) -> float:
        '''
        Compute F1 score between prediction and ground truth(s).

        Args:
            prediction: Predicted answer
            ground_truths: Ground truth answer(s), can be string or list of strings

        Returns:
            Maximum F1 score across all ground truths
        '''
        if prediction is None or ground_truths is None:
            return 0.0

        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        max_f1 = 0.0

        for ground_truth in ground_truths:
            if ground_truth is None:
                continue

            normalized_prediction = self.normalize_answer(prediction)
            normalized_ground_truth = self.normalize_answer(ground_truth)

            # Handle special cases for yes/no/noanswer
            if (normalized_prediction in ['yes', 'no', 'noanswer']
                    or normalized_ground_truth in ['yes', 'no', 'noanswer']):
                if normalized_prediction != normalized_ground_truth:
                    continue

            pred_tokens = normalized_prediction.split()
            gold_tokens = normalized_ground_truth.split()
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0

            if precision + recall == 0:
                continue

            f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)

        return max_f1

    def forward(self, data: dict) -> dict:
        '''
        Evaluate F1 score for a single data item.
        单条数据处理，框架自动多进程并发。

        Args:
            data: dict containing prediction and ground truth

        Returns:
            dict with F1 score added
        '''
        prediction = data.get(self.prediction_key, None)
        ground_truths = data.get(self.ground_truth_key, None)
        score = self.compute_f1(prediction, ground_truths)
        data[self.output_key] = score
        return data
