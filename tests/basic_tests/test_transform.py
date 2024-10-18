import lazyllm
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.doc_node import DocNode


class TestSentenceSplitter:
    def setup_method(self):
        """Setup for tests: initialize the SentenceSplitter."""
        self.splitter = SentenceSplitter(chunk_size=30, chunk_overlap=10)

    def test_forward(self):
        text = """ Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep."""  # noqa: E501
        docs = [DocNode(text=text)]

        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            "Before college the two main things I worked on, outside of school, were writing and programming.I didn't write essays.",  # noqa: E501
            "I didn't write essays.I wrote what beginning writers were supposed to write then, and probably still are: short stories.My stories were awful.",  # noqa: E501
            "My stories were awful.They had hardly any plot, just characters with strong feelings, which I imagined made them deep.",  # noqa: E501
        ]
        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts
