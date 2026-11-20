import pytest

from lazyllm import TrainableModule
from lazyllm.tools import ParameterExtractor


@pytest.fixture
def extractor():
    model = TrainableModule(trust_remote_code=False)
    return ParameterExtractor(model, ['age', 'name'], ['int', 'str'], ['Age', 'Name'], [True, False])


@pytest.mark.parametrize('value, expected', [('25', 25), ('-7', -7), (' 42 ', 42), (25, 25)])
def test_integer_values(extractor, value, expected):
    result = {'age': value, 'name': '007'}
    extractor.check_int_value(result)
    assert result == {'age': expected, 'name': '007'}
    assert type(result['age']) is int


@pytest.mark.parametrize('value', ['unknown', '3.5', None, []])
def test_unconvertible_values_remain_unchanged(extractor, value):
    result = {'age': value}
    extractor.check_int_value(result)
    assert result == {'age': value}


def test_missing_integer_is_not_added(extractor):
    result = {'name': '007'}
    extractor.check_int_value(result)
    assert result == {'name': '007'}
