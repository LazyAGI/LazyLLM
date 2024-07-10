from lazyllm.tools.rag.doc_impl import DocImplV2
from lazyllm.tools.rag.transform import SentenceSplitter
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
        assert "CoarseChunk" in self.doc_impl.node_groups
        assert "MediumChunk" in self.doc_impl.node_groups
        assert "FineChunk" in self.doc_impl.node_groups

    def test_create_node_group(self):
        self.doc_impl.create_node_group(
            name="CustomChunk",
            transform=SentenceSplitter,
            chunk_size=512,
            chunk_overlap=50,
        )
        assert "CustomChunk" in self.doc_impl.node_groups
        node_group = self.doc_impl.node_groups["CustomChunk"]
        assert node_group["transform"] == SentenceSplitter
        assert node_group["transform_kwargs"]["chunk_size"] == 512
        assert node_group["transform_kwargs"]["chunk_overlap"] == 50

    def test_retrieve(self):
        self.mock_embed.return_value = "[0.1, 0.2, 0.3]"
        result = self.doc_impl.retrieve(
            query="test query",
            group_name="FineChunk",
            similarity="dummy",
            index=None,
            topk=1,
            similarity_kws={},
        )
        node = result[0]
        assert node.text == "dummy text"
