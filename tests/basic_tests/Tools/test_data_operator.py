import os
import shutil
import tempfile
import time
import pytest
import random
import json
from lazyllm import config, LOG
from lazyllm.tools.data import demo1, demo2, pt, pt_mm, pt_text, data_register
from lazyllm.thirdparty import PIL


class MockModel:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    def __call__(self, string, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self


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
        func = demo2.AddSuffix(input_key='text', suffix='!!!', _max_workers=32, _concurrency_mode='process')
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

    @staticmethod
    def _test_image_file(name):
        data_path = config['data_path']
        return os.path.join(data_path, 'ci_data', name)

    def test_resolution_filter(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        op = pt_mm.resolution_filter(min_width=1, min_height=1, max_width=700, max_height=500)
        inputs = [
            {'image_path': ji, 'id': 1},
            {'image_path': '/nonexistent/path.png', 'id': 2},
        ]
        res = op(inputs)
        assert len(res) == 0

    def test_resolution_resize(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        with tempfile.TemporaryDirectory() as tmpdir:
            ji_tmp = os.path.join(tmpdir, 'ji.jpg')
            shutil.copy(ji, ji_tmp)
            op = pt_mm.resolution_resize(max_side=400, inplace=False)
            res = op([{'image_path': ji_tmp}])
            assert res and res[0].get('image_path')
            resized_path = res[0]['image_path'][0]
            assert resized_path != ji_tmp
            assert '_resized' in resized_path
            assert os.path.exists(ji)
            with PIL.Image.open(resized_path) as img:
                w, h = img.size
                assert max(w, h) <= 400

    def test_integrity_check(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        op = pt_mm.integrity_check()
        res = op([
            {'image_path': ji, 'id': 1},
            {'image_path': '/nonexistent/path.png', 'id': 2},
        ])
        assert len(res) == 1
        assert res[0]['id'] == 1
        assert res[0]['image_path'] == [ji]

    def test_image_dedup(self):
        ji = self._test_image_file('ji.jpg')
        dog = self._test_image_file('dog.png')
        if not os.path.exists(ji) or not os.path.exists(dog):
            pytest.skip('ci_data/ji.jpg or ci_data/dog.png not found')
        op = pt_mm.ImageDedup()
        batch = [
            {'image_path': ji, 'id': 1},
            {'image_path': ji, 'id': 2},
            {'image_path': dog, 'id': 3},
        ]
        res = op(batch)
        assert len(res) == 2
        assert {r['id'] for r in res} == {1, 3}

    def test_graph_retriever(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        op = pt_mm.GraphRetriever(context_key='context', img_key='img')
        data = {'context': 'Test content with {braces}', 'img': f'![]({ji})'}
        res = op([data])
        assert res and res[0]['context'] == 'Test content with {{braces}}'
        assert 'img' in res[0] and len(res[0]['img']) == 1
        assert os.path.isabs(res[0]['img'][0])

    def test_text_relevance_filter(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        expected_response = {'relevance': 0.9, 'reason': 'relevant'}
        vlm = MockModel(expected_response)
        op = pt_mm.TextRelevanceFilter(vlm, threshold=0.5, _concurrency_mode='single')
        inputs = [{'image_path': ji, 'text': 'a red image'}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['image_path'] == [ji]
        assert res[0]['image_text_relevance'] == 0.9

    def test_vqa_generator(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        expected_response = {
            'qa_pairs': [{'query': 'Q1', 'answer': 'A1'}, {'query': 'Q2', 'answer': 'A2'}],
        }
        vlm = MockModel(expected_response)
        op = pt_mm.VQAGenerator(vlm, num_qa=2, _concurrency_mode='single')
        inputs = [{'image_path': ji, 'context': 'A simple image.'}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['qa_pairs'] == [{'query': 'Q1', 'answer': 'A1'}, {'query': 'Q2', 'answer': 'A2'}]

    def test_phi4_qa_generator(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        expected_response = {
            'qa_pairs': [{'query': 'What is it?', 'answer': 'An image.'}],
        }
        vlm = MockModel(expected_response)
        op = pt_text.Phi4QAGenerator(vlm, num_qa=2, _concurrency_mode='single')
        inputs = [{'context': 'Some context.', 'image_path': ji}]
        res = op(inputs)
        assert len(res) == 1
        assert len(res[0]['qa_pairs']) == 1
        assert res[0]['qa_pairs'][0]['query'] == 'What is it?'

    def test_vqa_scorer(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        expected_response = {'score': 0.85, 'clarity': 0.9, 'composition': 0.8, 'reason': 'Good quality'}
        vlm = MockModel(expected_response)
        op = pt_mm.VQAScorer(vlm, _concurrency_mode='single')
        inputs = [{'image_path': ji}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['quality_score']['score'] == 0.85
        assert res[0]['quality_score']['clarity'] == 0.9
        assert res[0]['quality_score']['composition'] == 0.8

    def test_context_qual_filter(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        expected_response = {'score': 1, 'reason': 'suitable'}
        vlm = MockModel(expected_response)
        op = pt.ContextQualFilter(vlm, _concurrency_mode='single')
        inputs = [{'context': 'Good context for QA.', 'image_path': ji}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['context'] == 'Good context for QA.'

        expected_response_reject = {'score': 0, 'reason': 'not suitable'}
        vlm_reject = MockModel(expected_response_reject)
        op_reject = pt.ContextQualFilter(vlm_reject, _concurrency_mode='single')
        res_reject = op_reject(inputs)
        assert len(res_reject) == 0
