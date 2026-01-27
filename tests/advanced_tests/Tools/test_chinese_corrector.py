import unittest
import lazyllm
from lazyllm.tools.review.tools.chinese_corrector import ChineseCorrector
from lazyllm.launcher import cleanup


class TestChineseCorrector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = lazyllm.TrainableModule('Qwen2.5-32B-Instruct').start()
        cls.corrector = ChineseCorrector(llm=cls.model)

    @classmethod
    def tearDownClass(cls):
        cleanup()

    def test_correct_single_sentence(self):

        sentence = '我喜欢学习编成，但是有时候会感到困难。'
        result = self.corrector.correct(sentence)

        assert isinstance(result, dict)
        assert 'source' in result
        assert 'target' in result
        assert 'errors' in result
        assert result['source'] == sentence
        assert isinstance(result['target'], str)
        assert isinstance(result['errors'], list)

    def test_correct_sentence_with_typos(self):

        sentence = '昨天我去了一家很好次餐厅吃饭。'
        result = self.corrector.correct(sentence)
        assert isinstance(result, dict)
        assert result['source'] == sentence
        assert isinstance(result['target'], str)
        assert isinstance(result['errors'], list)

    def test_correct_empty_sentence(self):

        result = self.corrector.correct('')

        assert result['source'] == ''
        assert result['target'] == ''
        assert result['errors'] == []

    def test_correct_batch_empty_list(self):
        results = self.corrector.correct_batch([])

        assert results == []
