import lazyllm
from lazyllm.tools.rag.doc_impl import DocImpl
from lazyllm.tools.rag.store.store_base import LAZY_IMAGE_GROUP
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.store import LAZY_ROOT_NAME
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_PATH, RAG_DOC_ID
from lazyllm.tools.rag import Document, Retriever, TransformArgs, AdaptiveTransform
from lazyllm.tools.rag.utils import gen_docid
from lazyllm.launcher import cleanup
from unittest.mock import MagicMock
import unittest
import os
import time
import tempfile


class TestDocImpl(unittest.TestCase):

    def setUp(self):
        self.mock_embed = MagicMock()
        self.mock_directory_reader = MagicMock()
        # use temporary file as only existing files can be added to DocImpl
        self.tmp_file_a = tempfile.NamedTemporaryFile()
        self.tmp_file_b = tempfile.NamedTemporaryFile()
        mock_node = DocNode(group=LAZY_ROOT_NAME, text='dummy text')
        mock_node._global_metadata = {RAG_DOC_ID: gen_docid(self.tmp_file_a.name), RAG_DOC_PATH: self.tmp_file_a.name}
        self.mock_directory_reader.load_data.return_value = {LAZY_ROOT_NAME: [mock_node], LAZY_IMAGE_GROUP: []}

        self.doc_impl = DocImpl(embed=self.mock_embed, doc_files=[self.tmp_file_a.name])
        self.doc_impl._reader = self.mock_directory_reader

    def tearDown(self):
        self.tmp_file_a.close()
        self.tmp_file_b.close()

    def _build_root_nodes(self, input_files):
        root_nodes = []
        for path in input_files:
            node = DocNode(group=LAZY_ROOT_NAME, text=os.path.basename(path))
            node._global_metadata = {RAG_DOC_ID: gen_docid(path), RAG_DOC_PATH: path}
            root_nodes.append(node)
        return {LAZY_ROOT_NAME: root_nodes, LAZY_IMAGE_GROUP: []}

    def test_create_node_group_default(self):
        self.doc_impl._create_builtin_node_group('MyChunk', transform=lambda x: ','.split(x))
        self.doc_impl._lazy_init()
        assert 'MyChunk' in self.doc_impl.node_groups
        assert 'CoarseChunk' in self.doc_impl.node_groups
        assert 'MediumChunk' in self.doc_impl.node_groups
        assert 'FineChunk' in self.doc_impl.node_groups

    def test_create_node_group(self):
        self.doc_impl._lazy_init.flag.reset()
        self.doc_impl.create_node_group(
            name='CustomChunk',
            transform=SentenceSplitter,
            chunk_size=512,
            chunk_overlap=50,
        )
        assert 'CustomChunk' in self.doc_impl.node_groups
        node_group = self.doc_impl.node_groups['CustomChunk']
        assert node_group['transform'].f == SentenceSplitter
        assert node_group['transform'].kwargs['chunk_size'] == 512
        assert node_group['transform']['kwargs']['chunk_overlap'] == 50

    def test_retrieve(self):
        self.mock_embed.return_value = '[0.1, 0.2, 0.3]'
        self.doc_impl.activate_group(Document.FineChunk, [])
        result = self.doc_impl.retrieve(
            query='test query',
            group_name='FineChunk',
            similarity='bm25',
            similarity_cut_off=-100,
            index='default',
            topk=1,
            similarity_kws={},
        )
        node = result[0]
        assert node.text == 'dummy text'

    def test_add_files(self):
        assert self.doc_impl._store is None
        self.doc_impl._lazy_init()
        assert len(self.doc_impl.store.get_nodes(group=LAZY_ROOT_NAME)) == 1
        new_doc = DocNode(text='new dummy text', group=LAZY_ROOT_NAME)
        new_doc._global_metadata = {RAG_DOC_ID: gen_docid(self.tmp_file_b.name), RAG_DOC_PATH: self.tmp_file_b.name}
        self.mock_directory_reader.load_data.return_value = {LAZY_ROOT_NAME: [new_doc], LAZY_IMAGE_GROUP: []}
        self.doc_impl._processor.add_doc([self.tmp_file_b.name])
        assert len(self.doc_impl.store.get_nodes(group=LAZY_ROOT_NAME)) == 2

    def test_dataset_path_sync_without_doc_list_manager(self):
        self.mock_embed.return_value = [0.1, 0.2, 0.3]
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test.txt')
            with open(file_path, 'w') as file:
                file.write('local dataset path')

            doc_impl = DocImpl(embed=self.mock_embed, dataset_path=temp_dir, enable_path_monitoring=False)
            doc_impl._reader = MagicMock()
            doc_impl._reader.load_data.side_effect = (
                lambda input_files, metadatas, split_nodes_by_type=True: self._build_root_nodes(input_files)
            )
            doc_impl._lazy_init()

            nodes = doc_impl.store.get_nodes(group=LAZY_ROOT_NAME)
            assert len(nodes) == 1
            assert nodes[0].global_metadata[RAG_DOC_ID] == gen_docid(file_path)
            assert nodes[0].global_metadata[RAG_DOC_PATH] == file_path
            assert not hasattr(doc_impl, '_dlm')

    def test_dataset_path_monitor_adds_and_removes_files(self):
        self.mock_embed.return_value = [0.1, 0.2, 0.3]
        with tempfile.TemporaryDirectory() as temp_dir:
            file_a = os.path.join(temp_dir, 'a.txt')
            with open(file_a, 'w') as file:
                file.write('a')

            doc_impl = DocImpl(embed=self.mock_embed, dataset_path=temp_dir, enable_path_monitoring=True)
            doc_impl._reader = MagicMock()
            doc_impl._reader.load_data.side_effect = (
                lambda input_files, metadatas, split_nodes_by_type=True: self._build_root_nodes(input_files)
            )
            doc_impl._local_monitor_interval = 0.1
            doc_impl._lazy_init()

            def wait_for_doc_ids(expected_ids):
                deadline = time.time() + 3
                while time.time() < deadline:
                    nodes = doc_impl.store.get_nodes(group=LAZY_ROOT_NAME)
                    current_ids = {node.global_metadata[RAG_DOC_ID] for node in nodes}
                    if current_ids == expected_ids:
                        return
                    time.sleep(0.1)
                raise AssertionError(f'Expected doc ids {expected_ids}, got {current_ids}')

            try:
                wait_for_doc_ids({gen_docid(file_a)})

                file_b = os.path.join(temp_dir, 'b.txt')
                with open(file_b, 'w') as file:
                    file.write('b')
                wait_for_doc_ids({gen_docid(file_a), gen_docid(file_b)})

                os.remove(file_a)
                wait_for_doc_ids({gen_docid(file_b)})
            finally:
                doc_impl._local_monitor_continue = False
                if doc_impl._local_monitor_thread:
                    doc_impl._local_monitor_thread.join(timeout=1)

class TestDocument(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        cleanup()

    def test_register_global_and_local(self):
        Document.create_node_group('Chunk1', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
        Document.create_node_group('Chunk2', transform=TransformArgs(
            f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        doc1, doc2 = Document('rag_master'), Document('rag_master')
        doc2.create_node_group('Chunk2', transform=dict(
            f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=10)))
        doc2.create_node_group('Chunk3', trans_node=True,
                               transform=lazyllm.pipeline(SentenceSplitter(chunk_size=128, chunk_overlap=10)))
        doc1._impl._lazy_init()
        doc2._impl._lazy_init()
        assert doc1._impl.node_groups['Chunk1']['transform']['kwargs']['chunk_size'] == 512
        assert doc1._impl.node_groups['Chunk2']['transform']['kwargs']['chunk_size'] == 256
        assert doc2._impl.node_groups['Chunk1']['transform']['kwargs']['chunk_size'] == 512
        assert doc2._impl.node_groups['Chunk2']['transform']['kwargs']['chunk_size'] == 128
        assert 'Chunk3' not in doc1._impl.node_groups
        assert isinstance(doc2._impl.node_groups['Chunk3']['transform']['f'], lazyllm.pipeline)
        assert doc2._impl.node_groups['Chunk3']['transform']['trans_node'] is True

        retriever = Retriever([doc1, doc2], 'Chunk2', similarity='bm25', topk=2)
        r = retriever('什么是道')
        assert isinstance(r, list)
        assert len(r) == 4
        assert isinstance(r[0], DocNode)

        retriever2 = Retriever([doc1, doc2], 'Chunk3', similarity='bm25', topk=2)
        r = retriever2('什么是道')
        assert isinstance(r, list)
        assert len(r) == 2
        assert isinstance(r[0], DocNode)

    def test_create_document(self):
        Document('rag_master')
        Document('rag_master/')

    def test_register_with_pattern(self):
        Document.create_node_group('AdaptiveChunk1', transform=[
            TransformArgs(f=SentenceSplitter, pattern='*.txt', kwargs=dict(chunk_size=512, chunk_overlap=50)),
            dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25))])
        Document.create_node_group('AdaptiveChunk2', transform=AdaptiveTransform([
            dict(f=SentenceSplitter, pattern=(lambda x: x.endswith('.txt')),
                 kwargs=dict(chunk_size=512, chunk_overlap=50)),
            TransformArgs(f=SentenceSplitter, pattern=None, kwargs=dict(chunk_size=256, chunk_overlap=25))]))
        doc = Document('rag_master')
        doc._impl._lazy_init()
        retriever = Retriever(doc, 'AdaptiveChunk1', similarity='bm25', topk=2)
        retriever('什么是道')
        retriever = Retriever(doc, 'AdaptiveChunk2', similarity='bm25', topk=2)
        retriever('什么是道')

    def test_create_node_group_with_ref(self):
        doc = Document('rag_master')
        # Create parent node group
        doc.create_node_group('parent_chunk', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
        # Create ref node group under parent
        doc.create_node_group('ref_chunk', parent='parent_chunk',
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=12)))

        # Create node group with ref - ref_chunk is descendant of parent_chunk
        def transform_with_ref(text, ref):
            return f'doc_test_text: {text}\n doc_test_ref:' + '\n'.join(ref)

        doc.create_node_group('chunk_with_ref', parent='parent_chunk',
                              transform=transform_with_ref, ref='ref_chunk')

        assert 'chunk_with_ref' in doc._impl.node_groups
        assert doc._impl.node_groups['chunk_with_ref']['ref'] == 'ref_chunk'

        doc.activate_groups(['chunk_with_ref'])
        doc.start()
        ref_nodes = doc._impl.store.get_nodes(group='chunk_with_ref')
        assert len(ref_nodes) > 0
        assert 'doc_test_text' in ref_nodes[0].text
        assert 'doc_test_ref' in ref_nodes[0].text

    def test_create_node_group_with_invalid_ref(self):
        doc = Document('rag_master')
        # Create two independent node groups
        doc.create_node_group('group_a', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
        doc.create_node_group('group_b', transform=SentenceSplitter, chunk_size=256, chunk_overlap=25)

        # Try to create node group with ref that is not descendant of parent - should raise error
        doc.create_node_group('invalid_ref_chunk', parent='group_a',
                              transform=lambda x: [x], ref='group_b')

        # Activate the group first, then init to trigger validation
        doc._impl._activated_groups.add('invalid_ref_chunk')
        with self.assertRaises(ValueError) as context:
            doc._impl._lazy_init()
        assert 'not under parent' in str(context.exception)

    def test_find(self):
        #       /- MediumChunk
        #      /                /- chunk1 -- chunk11 -- chunk111
        # root --- CoarseChunk <           /- chunk21
        #      \                \- chunk2 <
        #       \- FineChunk               \- chunk22
        doc = Document('rag_master')
        doc.create_node_group('chunk1', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        doc.create_node_group('chunk11', parent='chunk1',
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=16)))
        doc.create_node_group('chunk111', parent='chunk11',
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=64, chunk_overlap=12)))
        doc.create_node_group('chunk2', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        doc.create_node_group('chunk21', parent='chunk2',
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=64, chunk_overlap=8)))
        doc.create_node_group('chunk22', parent='chunk2',
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=64, chunk_overlap=8)))

        def _test_impl(group, target):
            retriever = Retriever(doc, group, similarity='bm25', topk=3, target=target)
            r = retriever('何为天道')
            assert r[0]._group == target or group, f'expect {target or group}, bug get {r[0]._group}'

        for group, target in [('chunk11', None), ('chunk11', 'chunk1'), (Document.CoarseChunk, 'chunk111'),
                              ('chunk11', 'chunk22'), ('chunk111', 'chunk21'), ('chunk1', 'chunk21'),
                              ('chunk111', 'chunk21'), ('chunk21', 'chunk1'), ('chunk22', Document.FineChunk)]:
            _test_impl(group, target)

    def test_doc_web_module(self):
        import time
        import requests
        doc = Document('rag_master', manager='ui')
        doc.create_kb_group(name='test_group')
        doc2 = Document('rag_master', manager=doc.manager, name='test_group2')
        doc.start()
        time.sleep(4)
        url = doc._manager._docweb.url
        response = requests.get(url)
        assert response.status_code == 200
        assert doc2._curr_group == 'test_group2'
        assert doc2.manager == doc.manager
        doc.stop()

    def test_get_nodes(self):
        doc = Document('rag_master')
        doc.create_node_group('chunk1', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        doc.activate_groups(groups=['chunk1'])
        nodes = doc.get_nodes(group='chunk1', numbers=[2])
        assert len(nodes) > 0
        for n in nodes:
            assert n.number == 2

    def test_get_window_nodes(self):
        doc = Document('rag_master')
        doc.create_node_group('chunk1', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=12)))
        doc.activate_groups(groups=['chunk1'])

        nodes = doc.get_nodes(group='chunk1', numbers=[1])
        node = nodes[0]
        assert node.number == 1
        # node.number is 1, so the window is [1, 2, 3, 4]
        window = doc.get_window_nodes(node, span=(-1, 3), merge=False)
        assert len(window) == 4
        assert window == sorted(window, key=lambda n: n.number)
        assert all(n.number in [1, 2, 3, 4] for n in window)

        merged = doc.get_window_nodes(node, span=(-1, 3), merge=True)
        assert isinstance(merged, DocNode)
        assert merged.group == node.group
        assert merged.text == '\n'.join([n.text for n in window])

        nodes = doc.get_nodes(group='chunk1', numbers=[2])
        node = nodes[0]
        assert node.number == 2
        # node.number is 2, so the window is [1, 2, 3, 4, 5]
        window = doc.get_window_nodes(node, span=(-1, 3), merge=False)
        assert len(window) == 5
        assert window == sorted(window, key=lambda n: n.number)
        assert all(n.number in [1, 2, 3, 4, 5] for n in window)
