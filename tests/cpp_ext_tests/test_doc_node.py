class TestDocNode:
    def setup_method(self):
        from lazyllm import lazyllm_cpp
        self.lazyllm_cpp = lazyllm_cpp

    def test_doc_node_set_get(self):
        node = self.lazyllm_cpp.DocNode()
        assert node.get_text() == ''
        node.set_text('hello')
        assert node.get_text() == 'hello'

        node2 = self.lazyllm_cpp.DocNode('world')
        assert node2.get_text() == 'world'
