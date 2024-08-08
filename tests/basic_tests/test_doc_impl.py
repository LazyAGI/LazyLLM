from lazyllm.tools.rag.doc_impl import DocImplV2
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.store import DocNode, LAZY_ROOT_NAME
from lazyllm import call_once
from unittest.mock import MagicMock
import unittest


class TestDocImplV2(unittest.TestCase):

    def setUp(self):
        self.mock_embed = MagicMock()
        self.mock_directory_reader = MagicMock()
        mock_node = DocNode(group=LAZY_ROOT_NAME, text="dummy text")
        mock_node.metadata = {"file_name": "dummy_file.txt"}
        self.mock_directory_reader.load_data.return_value = [mock_node]

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
            similarity="bm25",
            similarity_cut_off=-100,
            index=None,
            topk=1,
            similarity_kws={},
        )
        node = result[0]
        assert node.text == "dummy text"

    def test_add_files(self):
        assert self.doc_impl.store is None
        call_once(self.doc_impl.init_flag, self.doc_impl._lazy_init)
        assert len(self.doc_impl.store.traverse_nodes(LAZY_ROOT_NAME)) == 1
        new_doc = DocNode(text="new dummy text", group=LAZY_ROOT_NAME)
        new_doc.metadata = {"file_name": "new_file.txt"}
        self.mock_directory_reader.load_data.return_value = [new_doc]
        self.doc_impl.add_files(["new_file.txt"])
        assert len(self.doc_impl.store.traverse_nodes(LAZY_ROOT_NAME)) == 2

    def test_delete_files(self):
        self.doc_impl.delete_files(["dummy_file.txt"])
        assert len(self.doc_impl.store.traverse_nodes(LAZY_ROOT_NAME)) == 0


if __name__ == "__main__":
    unittest.main()
