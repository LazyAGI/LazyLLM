import os
import time
import pytest
import random
import shutil
import json
from lazyllm import config, LOG
from lazyllm.tools.data import demo1, demo2, data_register


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
        assert func._concurrency_mode == 'thread'
        LOG.info(f'Max workers: {func._max_workers}')
        inputs = [{'text': text} for text in (['hello', 'world'] * 2000)]
        res = func(inputs)
        expected = [{'text': text.upper()} for text in (['hello', 'world'] * 2000)]
        assert sorted(res, key=lambda x: x['text']) == sorted(expected, key=lambda x: x['text'])

    def test_add_suffix(self):
        func = demo2.AddSuffix(input_key='text', suffix='!!!', _max_workers=64, _concurrency_mode='process')
        assert func._concurrency_mode == 'process'
        LOG.info(f'Max workers: {func._max_workers}')
        inputs = [{'text': text} for text in (['exciting', 'amazing'] * 2000)]
        res = func(inputs)
        expected = [{'text': text + '!!!'} for text in (['exciting', 'amazing'] * 2000)]
        assert sorted(res, key=lambda x: x['text']) == sorted(expected, key=lambda x: x['text'])

    def test_rich_content(self):
        func = demo2.rich_content(input_key='text')
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        assert res == [
            {'text': 'This is a test.'},
            {'text': 'This is a test. - part 1'},
            {'text': 'This is a test. - part 2'}]

    def test_output_file(self):
        func = demo2.rich_content(input_key='text').set_output(self.root_dir)
        inputs = [{'text': 'This is a test.'}]
        res = func(inputs)
        assert isinstance(res, str)
        assert os.path.exists(res)
        assert res.endswith('.jsonl')

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

    def test_process_safety_stress(self):
        # Test if multiple workers/instances cause file corruption or data loss
        count = 1000
        inputs = [{'text': f'id_{i}'} for i in range(count)]
        # Use a high number of workers to increase contention probability
        op = demo2.AddSuffix(input_key='text', suffix='_safe', _max_workers=32, _concurrency_mode='process')

        res = op(inputs)

        # 1. Check results count
        assert len(res) == count

        # 2. Check file integrity (no partial JSON writes)
        load_res = []
        with open(op._store.save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == count
            for line in lines:
                # If json.loads fails, it means process safety failed on IO
                data = json.loads(line)
                assert data['text'].endswith('_safe')
                load_res.append(data)
        # 3. Check all expected entries are present
        sorted_res = sorted(res, key=lambda x: x['text'])
        sorted_load = sorted(load_res, key=lambda x: x['text'])
        assert sorted_res == sorted_load

    @pytest.mark.skip(reason='Long running test')
    def test_dummy_llm_operator(self):
        num_qa = 60000

        @data_register('data.demo1', rewrite_func='forward', _concurrency_mode='thread')
        def dummy_llm_op(data, input_key='text', output_key='llm_output'):
            assert isinstance(data, dict)
            content = data.get(input_key, '')
            time.sleep(random.uniform(2, 12))  # Simulate LLM latency with variability
            data[output_key] = f'LLM response for: {content}'
            return data

        llm_func = demo1.dummy_llm_op(input_key='text', output_key='llm_output')
        assert llm_func._concurrency_mode == 'thread'
        inputs = [{'text': f'query_{i}', 'id': i} for i in range(num_qa)]
        res = llm_func(inputs)

        assert len(res) == num_qa

        sorted_res = sorted(res, key=lambda x: x['id'])
        for i, item in enumerate(sorted_res):
            expected_text = f'query_{i}'
            expected_llm = f'LLM response for: {expected_text}'
            assert item['text'] == expected_text
            assert item['llm_output'] == expected_llm

        load_res = []
        with open(llm_func._store.save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == num_qa
            for line in lines:
                data = json.loads(line)
                load_res.append(data)
        sorted_load = sorted(load_res, key=lambda x: x['id'])
        assert sorted_res == sorted_load
