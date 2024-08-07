from lazyllm.tools.rag.doc_impl import DocImplV2
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.store import DocNode, LAZY_ROOT_NAME
from unittest.mock import MagicMock
import unittest


class TestDocImplV2(unittest.TestCase):

    def setUp(self):
        self.mock_embed = MagicMock()
        self.mock_directory_reader = MagicMock()
        self.mock_directory_reader.load_data.return_value = [
            DocNode(uid="1", text="dummy text")
        ]

        self.doc_impl = DocImplV2(embed=self.mock_embed, doc_files=["dummy_file.txt"])
        self.doc_impl.directory_reader = self.mock_directory_reader
        self.doc_impl._lazy_init()

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
            similarity="bm25",
            similarity_cut_off=-100,
            index=None,
            topk=1,
            similarity_kws={},
        )
        node = result[0]
        assert node.text == "dummy text"

    def test_add_files(self):
        new_docs = [DocNode(text="new dummy text")]
        self.mock_directory_reader.load_data.return_value = new_docs
        self.doc_impl.add_files(["new_file.txt"])
        assert self.doc_impl.store.has_nodes(LAZY_ROOT_NAME)
        assert len(self.doc_impl.store.traverse_nodes(LAZY_ROOT_NAME)) == 1

    def test_delete_files(self):
        nodes_to_delete = [DocNode(uid="1")]
        self.mock_directory_reader.get_nodes_by_files.return_value = nodes_to_delete
        self.doc_impl.delete_files(["dummy_file.txt"])

        self.mock_directory_reader.get_nodes_by_files.assert_called_once_with(
            ["dummy_file.txt"]
        )
        assert len(self.doc_impl.store.traverse_nodes(LAZY_ROOT_NAME)) == 0


if __name__ == "__main__":
    unittest.main()
