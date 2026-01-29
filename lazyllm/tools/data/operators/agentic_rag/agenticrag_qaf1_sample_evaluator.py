"""AgenticRAG QA F1 Sample Evaluator operator"""
import re
import string
from collections import Counter
from tqdm import tqdm
import pandas as pd
from lazyllm import LOG
from ...base_data import data_register
funcs = data_register.new_group('function')
classes = data_register.new_group('class')

class AgenticRAGQAF1SampleEvaluator(classes):
    """
    Evaluator for computing F1 scores between predicted answers and reference answers.
    用于评估预测答案与多个参考答案之间的 F1 分数
    """

    def __init__(self):
        LOG.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "用于评估预测答案与多个参考答案之间的 F1 分数"

    def normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(self, prediction: str, ground_truths) -> float:
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

            if normalized_prediction in ["yes", "no", "noanswer"] or normalized_ground_truth in ["yes", "no", "noanswer"]:
                if normalized_prediction != normalized_ground_truth:
                    continue

            pred_tokens = normalized_prediction.split()
            gold_tokens = normalized_ground_truth.split()
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)

        return max_f1

    def __call__(
            self,
            data,
            prediction_key: str = "refined_answer",
            ground_truth_key: str = "golden_doc_answer",
            output_key: str = "F1Score",
    ):
        """
        Evaluate F1 scores for the data.

        Args:
            data: List of dict or pandas DataFrame
            prediction_key: Key for predicted answers
            ground_truth_key: Key for ground truth answers
            output_key: Key for output F1 scores

        Returns:
            List of dict with F1 scores added
        """
        if isinstance(data, pd.DataFrame):
            dataframe = data
        else:
            dataframe = pd.DataFrame(data)

        LOG.info(f"Evaluating {output_key}...")
        f1_scores = []

        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="F1Scorer Evaluating..."):
            prediction = row.get(prediction_key, None)
            ground_truths = row.get(ground_truth_key, None)
            score = self.compute_f1(prediction, ground_truths)
            f1_scores.append(score)

        dataframe[output_key] = f1_scores
        LOG.info("Evaluation complete!")

        return dataframe.to_dict('records')

