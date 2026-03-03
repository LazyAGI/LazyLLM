import os
import shutil
import json
import tempfile

from lazyllm import config
from lazyllm.tools.data.pipelines.codegen_pipelines import build_codegen_pipeline


class MockModelCallable:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def __call__(self, x):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


class TestCodegenPipeline:

    class MockModel:
        def __init__(self, return_val=None):
            self.return_val = return_val

        def share(self): return self

        def prompt(self, system): return self

        def formatter(self, fmt):
            self._formatter = fmt
            return self

        def __call__(self, x, **kwargs):
            if callable(self.return_val):
                result = self.return_val(x)
            else:
                result = self.return_val
            if hasattr(self, '_formatter') and self._formatter is not None:
                try:
                    import json
                    result = json.loads(result) if isinstance(result, str) else result
                except Exception:
                    pass
            return result

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_codegen_pipeline_filter(self):
        responses = [
            'Write a function to add two numbers.\n```python\ndef add(a, b):\n    return a + b\n```',
            '```python\ndef add(a, b):\n    return a + b\n```',
            json.dumps({'score': 5, 'feedback': 'Incomplete code.'}),
            json.dumps({'score': 5, 'feedback': 'Incomplete code.'}),
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = build_codegen_pipeline(model=mock_model, input_key='messages', min_score=7, max_score=10)
        data = [{'messages': [{'role': 'user', 'content': 'Write a function to add two numbers.'}]}]
        res = ppl(data)

        assert len(res) == 0

    def test_codegen_pipeline_success(self):
        responses = [
            'Write a function to add two numbers.\n```python\ndef add(a, b):\n    return a + b\n```',
            '```python\ndef add(a, b):\n    return a + b\n```',
            json.dumps({'score': 8, 'feedback': 'Good code.'}),
            json.dumps({'score': 8, 'feedback': 'Good code.'}),
        ]

        mock_model = self.MockModel(return_val=MockModelCallable(responses))

        ppl = build_codegen_pipeline(model=mock_model, input_key='messages', min_score=7, max_score=10)
        data = [{'messages': [{'role': 'user', 'content': 'Write a function to add two numbers.'}]}]
        res = ppl(data)

        # CodeFeedbackFormatter now returns only {'instruction', 'input', 'output'}
        assert len(res) == 1
        assert 'instruction' in res[0]
        assert 'input' in res[0]
        assert 'output' in res[0]
        assert res[0]['instruction'] == 'Write a function to add two numbers.'
        assert res[0]['input'] == ''
        assert 'def add(a, b):' in res[0]['output']
        assert '专家反馈' in res[0]['output']
