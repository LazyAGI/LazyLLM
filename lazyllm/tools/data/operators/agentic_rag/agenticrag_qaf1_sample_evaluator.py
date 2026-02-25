import re
import string
from collections import Counter
from ...base_data import data_register
from lazyllm.common.registry import LazyLLMRegisterMetaClass

# Get or create agenticrag group
if 'agenticrag' in LazyLLMRegisterMetaClass.all_clses['data']:
    agenticrag = LazyLLMRegisterMetaClass.all_clses['data']['agenticrag'].base
else:
    agenticrag = data_register.new_group('agenticrag')


def _normalize_response(response: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(response))))


def _compute_f1_score(predicted: str, reference: str) -> float:
    # Handle special cases for yes/no/noanswer
    if predicted in ['yes', 'no', 'noanswer'] or reference in ['yes', 'no', 'noanswer']:
        if predicted != reference:
            return 0.0

    pred_tokens = predicted.split()
    gold_tokens = reference.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


@data_register('data.agenticrag', rewrite_func='forward', _concurrency_mode='process')
def qaf1_normalize_texts(data: dict,
                         predicted_key: str = 'refined_answer',
                         reference_key: str = 'golden_doc_answer') -> dict:
    prediction = data.get(predicted_key, None)
    ground_truths = data.get(reference_key, None)

    if prediction is None or ground_truths is None:
        data['_normalized_prediction'] = None
        data['_normalized_ground_truths'] = None
        return data

    # Normalize prediction
    data['_normalized_prediction'] = _normalize_response(str(prediction))

    # Normalize ground truths (handle both string and list)
    if isinstance(ground_truths, str):
        data['_normalized_ground_truths'] = [_normalize_response(str(ground_truths))]
    else:
        data['_normalized_ground_truths'] = [
            _normalize_response(str(gt)) for gt in ground_truths if gt is not None
        ]

    return data


@data_register('data.agenticrag', rewrite_func='forward', _concurrency_mode='process')
def qaf1_calculate_score(data: dict, result_key: str = 'F1Score') -> dict:
    normalized_prediction = data.get('_normalized_prediction', None)
    normalized_ground_truths = data.get('_normalized_ground_truths', None)

    if normalized_prediction is None or not normalized_ground_truths:
        data[result_key] = 0.0
    else:
        max_f1 = 0.0
        for normalized_ground_truth in normalized_ground_truths:
            f1 = _compute_f1_score(normalized_prediction, normalized_ground_truth)
            max_f1 = max(max_f1, f1)
        data[result_key] = max_f1

    # Clean up temporary fields
    data.pop('_normalized_prediction', None)
    data.pop('_normalized_ground_truths', None)

    return data
