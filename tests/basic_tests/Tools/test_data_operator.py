import os
import shutil
import json
from lazyllm import config
from lazyllm.tools.data import demo1, demo2


class TestDataOperators:

    def setup_method(self):
        self.root_dir = './test_data_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_build_pre_suffix(self):
        func = demo1.build_pre_suffix(input_key='text', prefix='Hello, ', suffix='!')
        inputs = [{'text': 'world'}, {'text': 'lazyLLM'}]
        res = func(inputs)
        assert res == [{'text': 'Hello, world!'}, {'text': 'Hello, lazyLLM!'}]

    def test_process_uppercase(self):
        func = demo1.process_uppercase(input_key='text', _concurrency_mode='thread')
        inputs = [{'text': text} for text in (['hello', 'world'] * 60)]
        res = func(inputs)
        expected = [{'text': text.upper()} for text in (['hello', 'world'] * 60)]
        assert sorted(res, key=lambda x: x['text']) == sorted(expected, key=lambda x: x['text'])

    def test_add_suffix(self):
        func = demo2.AddSuffix(input_key='text', suffix='!!!', _max_workers=2, _concurrency_mode='process')
        inputs = [{'text': text} for text in (['exciting', 'amazing'] * 60)]
        res = func(inputs)
        expected = [{'text': text + '!!!'} for text in (['exciting', 'amazing'] * 60)]
        assert sorted(res, key=lambda x: x['text']) == sorted(expected, key=lambda x: x['text'])

    def test_rich_content(self):
        func = demo2.rich_content(input_key='text')
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        assert res == [
            {'text': 'This is a test.'},
            {'text': 'This is a test. - part 1'},
            {'text': 'This is a test. - part 2'}]

    def test_error_handling(self):
        op = demo2.error_prone_op(input_key='text', _save_data=True, _concurrency_mode='single')
        inputs = [{'text': 'ok1'}, {'text': 'fail'}, {'text': 'ok2'}]
        res = op(inputs)

        # Check results - failure should be skipped in valid results
        assert len(res) == 2
        assert res[0]['text'] == 'Processed: ok1'
        assert res[1]['text'] == 'Processed: ok2'

        # Check error file
        err_file = op._store.error_path
        assert os.path.exists(err_file)
        with open(err_file, 'r', encoding='utf-8') as f:
            errs = [json.loads(line) for line in f]
            assert len(errs) == 1
            assert errs[0]['text'] == 'fail'
            assert 'Intentional error for testing.' in errs[0]['infer_error']
