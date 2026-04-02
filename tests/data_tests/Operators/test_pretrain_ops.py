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
        self.root_dir = tempfile.mkdtemp()
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

    def test_text2json(self):
        expected_response = {
            'triples': [
                {'subject': 'Alice', 'predicate': 'works at', 'object': 'Company X'},
            ],
        }
        llm = MockModel(expected_response)
        op = pt.Text2Json(llm, input_key='text', output_key='parsed', _concurrency_mode='single')
        inputs = [{'text': 'Alice works at Company X.'}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['parsed'] == expected_response
        assert res[0]['parsed']['triples'][0]['subject'] == 'Alice'

        empty_llm = MockModel({})
        op_empty = pt.Text2Json(empty_llm, _concurrency_mode='single')
        res_empty = op_empty([{'text': 'Some text.'}])
        assert len(res_empty) == 1
        assert res_empty[0]['parsed'] == {}

    def test_context_expansion(self):
        expanded_text = 'This is a much longer and more detailed context about the topic.'
        llm = MockModel(expanded_text)
        op = pt.ContextExpansion(llm, _concurrency_mode='single')

        inputs = [{'context': 'Short ctx.', 'question': 'What is it?', 'answer': 'A thing.'}]
        res = op(inputs)
        assert len(res) == 1
        assert res[0]['expanded_context'] == expanded_text
        assert res[0]['context'] == 'Short ctx.'
        assert res[0]['question'] == 'What is it?'
        assert res[0]['answer'] == 'A thing.'

        assert pt.ContextExpansion(llm, _concurrency_mode='single')(
            [{'context': '', 'question': 'Q', 'answer': 'A'}]
        ) == []
        assert pt.ContextExpansion(llm, _concurrency_mode='single')(
            [{'context': 'ctx', 'question': '', 'answer': 'A'}]
        ) == []
        assert pt.ContextExpansion(llm, _concurrency_mode='single')(
            [{'context': 'ctx', 'question': 'Q', 'answer': ''}]
        ) == []

        llm_bad = MockModel(None)
        op_bad = pt.ContextExpansion(llm_bad, _concurrency_mode='single')
        assert op_bad(inputs) == []

    def test_context_reconstruction(self):
        op = pt.context_reconstruction(num_distractors=2, seed=42)
        batch = [
            {'expanded_context': f'ctx_{i}', 'question': f'Q{i}', 'answer': f'A{i}'}
            for i in range(4)
        ]
        res = op(batch)
        assert len(res) == 4
        for i, item in enumerate(res):
            assert item['context'] == f'ctx_{i}'
            assert 'long_context' in item
            assert item['question'] == f'Q{i}'
            assert item['answer'] == f'A{i}'
            passages = item['long_context'].split('\n\n')
            assert len(passages) == 3
            assert f'ctx_{i}' in item['long_context']

        custom_key_op = pt.context_reconstruction(long_context_key='lc', num_distractors=2, seed=42)
        custom_key_res = custom_key_op(batch)
        assert len(custom_key_res) == 4
        assert 'lc' in custom_key_res[0]
        assert 'long_context' not in custom_key_res[0]

        small_batch = [
            {'expanded_context': 'ctx_A', 'question': 'QA', 'answer': 'AA'},
            {'expanded_context': 'ctx_B', 'question': 'QB', 'answer': 'AB'},
        ]
        res2 = op(small_batch)
        assert len(res2) == 2
        for item in res2:
            passages = item['long_context'].split('\n\n')
            assert len(passages) == 2

        incomplete = [
            {'expanded_context': 'ctx_X', 'question': '', 'answer': 'AX'},
            {'expanded_context': 'ctx_Y', 'question': 'QY', 'answer': 'AY'},
        ]
        res3 = op(incomplete)
        assert len(res3) == 1
        assert res3[0]['question'] == 'QY'

        deterministic_op = pt.context_reconstruction(num_distractors=2, seed=0)
        res_a = deterministic_op(batch)
        res_b = deterministic_op(batch)
        assert [r['long_context'] for r in res_a] == [r['long_context'] for r in res_b]
