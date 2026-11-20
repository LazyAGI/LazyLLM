from lazyllm.tools.utils import chat_history_to_str
from lazyllm.tools.classifier.intent_classifier import IntentClassifier


def test_dictionary_history_matches_paired_history():
    history = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
    assert chat_history_to_str(history) == chat_history_to_str([['question', 'answer']])


def test_dictionary_history_skips_other_roles():
    history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'question'}]
    assert chat_history_to_str(history) == 'human: question\n'


def test_dictionary_history_keeps_recent_turn_limit():
    history = [{'role': 'user', 'content': str(index)} for index in range(41)]
    expected = ''.join(f'human: {index}\n' for index in range(1, 41))
    assert chat_history_to_str(history) == expected


def test_query_appended_after_history():
    history = [{'role': 'assistant', 'content': 'previous'}]
    assert chat_history_to_str(history, 'next') == 'assistant: previous\nhuman: next\n'


def test_empty_and_paired_history_controls():
    assert chat_history_to_str() == ''
    assert chat_history_to_str([['question', 'answer']], 'next') == 'human: question\nassistant: answer\nhuman: next\n'


def test_actual_intent_prompt_hook_preserves_dictionary_content():
    classifier = IntentClassifier(None)
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
    fields, history, tools, label = classifier.intent_promt_hook('next', messages, label='label')
    assert fields['history_info'] == 'human: question\nassistant: answer\n'
    assert history == []
    assert tools is None
    assert label == 'label'
