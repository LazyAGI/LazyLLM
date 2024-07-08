from lazyllm.tools.rag.doc_impl import DocImplV2
from lazyllm.tools.rag.parser import SentenceSplitter
from lazyllm.tools.rag.store import DocNode
from unittest.mock import MagicMock

class TestDocImplV2:
    def setup_method(self):
        self.mock_embed = MagicMock()
        self.mock_directory_reader = MagicMock()
        self.mock_directory_reader.load_data.return_value = [DocNode(text="dummy text")]

        self.doc_impl = DocImplV2(embed=self.mock_embed, doc_files=["dummy_file.txt"])
        self.doc_impl.directory_reader = self.mock_directory_reader

    def test_create_node_group_default(self):
        assert "CoarseChunk" in self.doc_impl.parser_dict
        assert "MediumChunk" in self.doc_impl.parser_dict
        assert "FineChunk" in self.doc_impl.parser_dict
        assert "SentenceSplitter" in self.doc_impl.parser_dict

    def test_create_node_group(self):
        self.doc_impl.create_node_group(
            name="CustomChunk",
            transform=SentenceSplitter,
            chunk_size=512,
            chunk_overlap=50,
        )
        assert "CustomChunk" in self.doc_impl.parser_dict
        parser_info = self.doc_impl.parser_dict["CustomChunk"]
        assert parser_info["parser"] == SentenceSplitter
        assert parser_info["parser_kwargs"]["chunk_size"] == 512
        assert parser_info["parser_kwargs"]["chunk_overlap"] == 50

    def test_retrieve(self):
        self.mock_embed.return_value = "[0.1, 0.2, 0.3]"
        self.doc_impl.index.registered_similarity = {
            "default": (MagicMock(return_value=[]), False)
        }
        result = self.doc_impl.retrieve(query="test query", parser_name="FineChunk", similarity="default", index=None, topk=1, similarity_kws={})
        assert result == []
