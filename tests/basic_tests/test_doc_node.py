from lazyllm.tools.rag.store import DocNode, MetadataMode


class TestDocNode:
    def setup_method(self):
        """Setup for tests: initialize common test data."""
        self.text = "This is a test document."
        self.metadata = {"author": "John Doe", "date": "2023-07-01"}
        self.embedding = [0.1, 0.2, 0.3]
        self.node = DocNode(
            text=self.text,
            metadata=self.metadata,
            embedding=self.embedding,
            excluded_embed_metadata_keys=["author"],
            excluded_llm_metadata_keys=["date"],
        )

    def test_node_creation(self):
        """Test the creation of a DocNode."""
        assert self.node.text == self.text
        assert self.node.metadata == self.metadata
        assert self.node.embedding == self.embedding
        assert self.node.excluded_embed_metadata_keys == ["author"]
        assert self.node.excluded_llm_metadata_keys == ["date"]

    def test_get_content(self):
        """Test the get_content method."""
        content = self.node.get_content(metadata_mode=MetadataMode.NONE)
        assert content == self.text

        content_with_metadata = self.node.get_content(metadata_mode=MetadataMode.ALL)
        expected_content_set = {"author: John Doe", "date: 2023-07-01", self.text}
        for s in expected_content_set:
            assert s in content_with_metadata

    def test_get_metadata_str(self):
        """Test the get_metadata_str method."""
        metadata_str_all = self.node.get_metadata_str(mode=MetadataMode.ALL)
        expected_metadata_set = {"author: John Doe", "date: 2023-07-01"}
        assert set(metadata_str_all.split("\n")) == expected_metadata_set

        metadata_str_llm = self.node.get_metadata_str(mode=MetadataMode.LLM)
        expected_metadata_str_llm = {"author: John Doe"}
        assert set(metadata_str_llm.split("\n")) == expected_metadata_str_llm

        metadata_str_embed = self.node.get_metadata_str(mode=MetadataMode.EMBED)
        expected_metadata_str_embed = {"date: 2023-07-01"}
        assert set(metadata_str_embed.split("\n")) == expected_metadata_str_embed

        metadata_str_none = self.node.get_metadata_str(mode=MetadataMode.NONE)
        assert metadata_str_none == ""

    def test_get_embedding(self):
        """Test the get_embedding method."""
        embedding = self.node.get_embedding()
        assert embedding == self.embedding

    def test_root_node(self):
        """Test the root_node property."""
        child_node = DocNode(text="Child node", parent=self.node)
        assert child_node.root_node == self.node

    def test_metadata_property(self):
        """Test the metadata property getter and setter."""
        new_metadata = {"editor": "Jane Doe"}
        self.node.metadata = new_metadata
        assert self.node.metadata == new_metadata
