import os
import pytest
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data import pt, pt_mm
from lazyllm.thirdparty import PIL


class MockModel:
    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def __call__(self, string: str, **kwargs):
        return self.mock_response

    def prompt(self, prompt):
        return self

    def formatter(self, formatter):
        return self

    def share(self):
        return self

    def start(self):
        return self


class TestPretrainOperators:

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
        op = pt_mm.GraphRetriever(context_key='context', img_key='img', _save_data=False)
        data = {'context': f'Test content ![]({ji}) with {{braces}}'}
        res = op([data])
        assert res and 'Test content' in res[0]['context']
        assert 'img' in res[0] and len(res[0]['img']) == 1
        assert os.path.isabs(res[0]['img'][0])

        # empty context: data kept, image_path as []
        empty_res = op([{'context': '   ', 'id': 1}])
        assert len(empty_res) == 1
        assert empty_res[0]['context'] == '   '
        assert empty_res[0]['img'] == []

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
        op = pt.Phi4QAGenerator(vlm, num_qa=2, _concurrency_mode='single')
        inputs = [{'context': 'Some context.', 'image_path': ji}]
        res = op(inputs)
        assert len(res) == 1
        assert len(res[0]['qa_pairs']) == 1
        assert res[0]['qa_pairs'][0]['query'] == 'What is it?'

    def test_vqa_scorer(self):
        ji = self._test_image_file('ji.jpg')
        if not os.path.exists(ji):
            pytest.skip('ci_data/ji.jpg not found')
        expected_response = {
            'score': 0.85,
            'relevance': 0.9,
            'correctness': 0.8,
            'reason': 'Good VQA quality',
        }
        vlm = MockModel(expected_response)
        op = pt_mm.VQAScorer(vlm, _concurrency_mode='single')
        inputs = [{'image_path': ji, 'query': 'What color is it?', 'answer': 'Red'}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['quality_score']['score'] == 0.85
        assert res[0]['quality_score']['relevance'] == 0.9
        assert res[0]['quality_score']['correctness'] == 0.8

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
