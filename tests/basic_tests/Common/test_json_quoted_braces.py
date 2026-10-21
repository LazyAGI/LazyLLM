import json

import pytest

import lazyllm


@pytest.mark.parametrize('value', [
    '{', '}', 'a{b', 'a}b', 'balanced{}', 'quote " {',
    'path' + chr(92), 'path' + chr(92) * 2,
])
@pytest.mark.parametrize('wrapped', [False, True])
def test_valid_json_string_content(value, wrapped):
    expected = {'text': value}
    text = json.dumps(expected)
    if wrapped:
        text = f'Result: {text} done'
    assert lazyllm.JsonFormatter()(text) == expected


def test_array_control():
    assert lazyllm.JsonFormatter()('[{"n": 1}, {"n": 2}]') == [{'n': 1}, {'n': 2}]


def test_multiple_objects_control():
    assert lazyllm.JsonFormatter()('first {"n": 1} then {"n": 2}') == [{'n': 1}, {'n': 2}]


def test_actual_pipeline_formatter_slice():
    flow = lazyllm.pipeline(lambda value: value, lazyllm.JsonFormatter('[text]'))
    assert flow('{"text": "{"}') == '{'


def test_non_json_control():
    assert lazyllm.JsonFormatter()('not json') == ''
