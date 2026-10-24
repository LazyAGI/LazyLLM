import pytest

from lazyllm import TrainableModule
from lazyllm.tools.data.pipelines.codegen_pipelines import build_codegen_pipeline


@pytest.fixture(scope='module')
def model():
    return TrainableModule(trust_remote_code=False)


@pytest.mark.parametrize('input_key', ['messages', 'prompt', 'custom_messages'])
@pytest.mark.parametrize('source', [
    'original request',
    {'role': 'user', 'content': 'original request'},
    [{'role': 'system', 'content': 'system context'}, {'role': 'human', 'content': 'original request'}],
])
def test_built_formatter_uses_configured_original_input(model, input_key, source):
    operation = build_codegen_pipeline(model, input_key=input_key)
    formatter = operation.code_feedback_formatter
    formatter._max_workers = 1
    row = {input_key: source, 'instruction': 'standardized request',
           'new_code': 'def solution(): return 1', 'feedback': 'accepted', 'quality_score': 9}
    result = formatter([row])
    assert result == [{'instruction': 'original request', 'input': '', 'output': 'def solution(): return 1'}]
