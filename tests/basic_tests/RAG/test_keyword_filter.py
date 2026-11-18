from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.rerank import Reranker


class TestKeywordFilter(object):

    def setup_method(self):
        self.nodes = [DocNode(text=f'This note is about {fruit}.') for fruit in ('apple', 'banana', 'cherry')]

    def _texts(self, reranker):
        return [node.get_text() for node in reranker(self.nodes)]

    def test_required_keys_are_isolated_between_filters(self):
        apple = Reranker('KeywordFilter', required_keys=['apple'])
        banana = Reranker('KeywordFilter', required_keys=['banana'])
        assert self._texts(apple) == ['This note is about apple.']
        assert self._texts(banana) == ['This note is about banana.']
        assert self._texts(apple) == ['This note is about apple.']

    def test_exclude_keys_are_isolated_between_filters(self):
        no_cherry = Reranker('KeywordFilter', exclude_keys=['cherry'])
        no_apple = Reranker('KeywordFilter', exclude_keys=['apple'])
        assert self._texts(no_cherry) == ['This note is about apple.', 'This note is about banana.']
        assert self._texts(no_apple) == ['This note is about banana.', 'This note is about cherry.']
        assert self._texts(no_cherry) == ['This note is about apple.', 'This note is about banana.']
