import lazyllm
import cloudpickle
from lazyllm.tools.rag.doc_impl import DocImpl
from lazyllm.tools.rag.store.store_base import LAZY_IMAGE_GROUP
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.store import LAZY_ROOT_NAME
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_PATH, RAG_DOC_ID
from lazyllm.tools.rag import Document, Retriever, TransformArgs, AdaptiveTransform
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

import lazyllm.tools.rag.document as document_module
from lazyllm.launcher import cleanup
from lazyllm.tools.rag.utils import gen_docid


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

    def test_doc_impl_can_be_pickled_before_lazy_init(self):
        doc_impl = DocImpl(embed=self.mock_embed, doc_files=[self.tmp_file_a.name])
        serialized = cloudpickle.dumps(doc_impl)
        restored = cloudpickle.loads(serialized)

        assert restored._local_monitor_lock is not None
        assert restored._local_monitor_thread is None
        assert restored._local_monitor_continue is False

    def test_doc_impl_resets_local_monitor_state_on_pickle(self):
        doc_impl = DocImpl(embed=self.mock_embed, doc_files=[self.tmp_file_a.name])
        doc_impl._local_monitor_thread = threading.Thread(target=lambda: None)
        doc_impl._local_monitor_continue = True

        serialized = cloudpickle.dumps(doc_impl)
        restored = cloudpickle.loads(serialized)

        assert restored._local_monitor_lock is not None
        assert restored._local_monitor_thread is None
        assert restored._local_monitor_continue is False

class TestDocument(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        cleanup()

    def setUp(self):
        self._temp_dirs = []

    def tearDown(self):
        for temp_dir in self._temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_dataset(self, text: str = None) -> str:
        temp_dir = tempfile.mkdtemp(prefix='lazyllm_document_')
        self._temp_dirs.append(temp_dir)
        file_path = os.path.join(temp_dir, 'rag.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text or '\n'.join(
                f'第{i}段：何为天道？人法地，地法天，天法道，道法自然。什么是道？道可道，非常道。'
                for i in range(1, 241)
            ))
        return temp_dir

    def test_register_global_and_local(self):
        Document.create_node_group('Chunk1', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
        Document.create_node_group('Chunk2', transform=TransformArgs(
            f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        dataset_path = self._build_dataset()
        doc1, doc2 = Document(dataset_path), Document(dataset_path)
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
        dataset_path = self._build_dataset()
        Document(dataset_path)
        Document(dataset_path + os.sep)

    def test_dataset_path_enables_monitoring_by_default_without_manager(self):
        doc = Document(self._build_dataset())
        assert doc._manager._enable_path_monitoring is True
        assert doc._impl._dataset_path == doc._manager._origin_path

    def test_register_with_pattern(self):
        Document.create_node_group('AdaptiveChunk1', transform=[
            TransformArgs(f=SentenceSplitter, pattern='*.txt', kwargs=dict(chunk_size=512, chunk_overlap=50)),
            dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25))])
        Document.create_node_group('AdaptiveChunk2', transform=AdaptiveTransform([
            dict(f=SentenceSplitter, pattern=(lambda x: x.endswith('.txt')),
                 kwargs=dict(chunk_size=512, chunk_overlap=50)),
            TransformArgs(f=SentenceSplitter, pattern=None, kwargs=dict(chunk_size=256, chunk_overlap=25))]))
        doc = Document(self._build_dataset())
        doc._impl._lazy_init()
        retriever = Retriever(doc, 'AdaptiveChunk1', similarity='bm25', topk=2)
        retriever('什么是道')
        retriever = Retriever(doc, 'AdaptiveChunk2', similarity='bm25', topk=2)
        retriever('什么是道')

    def test_create_node_group_with_ref(self):
        doc = Document(self._build_dataset())
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
        doc = Document(self._build_dataset())
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
        doc = Document(self._build_dataset())
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
        dataset_path = self._build_dataset()
        doc = Document(dataset_path, manager=True, create_ui=True)
        try:
            doc.start()
            doc.create_kb_group(name='test_group')
            doc2 = Document(dataset_path, manager=doc.manager, name='test_group2')
            assert hasattr(doc._manager, '_docweb')
            assert doc2._curr_group == 'test_group2'
            assert doc2.manager == doc.manager
        finally:
            doc.stop()

    def test_manager_ui_remains_compatible(self):
        dataset_path = self._build_dataset()
        doc = Document(dataset_path, manager='ui')
        try:
            doc.start()
            assert hasattr(doc._manager, '_docweb')
            assert doc._manager._enable_path_monitoring is False
        finally:
            doc.stop()

    def test_create_ui_requires_doc_server(self):
        with self.assertRaisesRegex(ValueError, 'requires an available DocServer'):
            Document(self._build_dataset(), create_ui=True)

    def test_document_processor_manager_constraints(self):
        dataset_path = self._build_dataset()
        processor = document_module.DocumentProcessor(url='http://127.0.0.1:9966')

        with self.assertRaisesRegex(ValueError, 'store_conf'):
            Document(dataset_path, manager=processor)
        with self.assertRaisesRegex(ValueError, 'pure local map store'):
            Document(dataset_path, manager=processor, store_conf={'type': 'map'})

        doc = Document(
            dataset_path,
            manager=processor,
            store_conf={'type': 'milvus', 'kwargs': {'uri': 'http://localhost:19530'}},
        )
        try:
            assert doc._manager._enable_path_monitoring is False
            assert doc._impl._enable_path_monitoring is False
            assert doc._impl._dataset_path == doc._manager._origin_path
        finally:
            doc.stop()

    def test_get_nodes(self):
        doc = Document(self._build_dataset())
        doc.create_node_group('chunk1', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        doc.activate_groups(groups=['chunk1'])
        nodes = doc.get_nodes(group='chunk1', numbers=[2])
        assert len(nodes) > 0
        for n in nodes:
            assert n.number == 2

    def test_get_window_nodes(self):
        doc = Document(self._build_dataset())
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

    def test_local_failed_insert_not_tracked(self):
        """Regression: _add_doc_to_store failures must not be tracked in _tracked_docs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_a = os.path.join(temp_dir, 'a.txt')
            file_b = os.path.join(temp_dir, 'b.txt')
            with open(file_a, 'w') as f:
                f.write('a')
            with open(file_b, 'w') as f:
                f.write('b')

            mock_embed = MagicMock()
            mock_embed.return_value = [0.1, 0.2, 0.3]
            doc_impl = DocImpl(embed=mock_embed, dataset_path=temp_dir, enable_path_monitoring=False)

            call_count = [0]
            original_add = doc_impl._processor.add_doc if hasattr(doc_impl, '_processor') and doc_impl._processor else None

            def mock_add_doc(files, ids, metadatas=None):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError('Simulated add failure')
                from lazyllm.tools.rag.doc_node import DocNode
                from lazyllm.tools.rag.store import LAZY_ROOT_NAME
                from lazyllm.tools.rag.store.store_base import LAZY_IMAGE_GROUP
                from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_DOC_PATH
                node = DocNode(group=LAZY_ROOT_NAME, text='test')
                node._global_metadata = {RAG_DOC_ID: ids[0], RAG_DOC_PATH: files[0]}
                return {LAZY_ROOT_NAME: [node], LAZY_IMAGE_GROUP: []}

            doc_impl._reader = MagicMock()
            doc_impl._reader.load_data.side_effect = (
                lambda input_files, metadatas, split_nodes_by_type=True: mock_add_doc(input_files, [f'id_{i}' for i in range(len(input_files))], metadatas)
            )
            doc_impl._lazy_init()

            # Only 1 out of 2 files should be tracked (the other failed)
            assert len(doc_impl._tracked_docs) == 1

    def test_persistent_store_auto_upgrade_nonexisting_dir(self):
        """Regression: auto-upgrade must trigger for non-existing dataset_path (future dir).

        Instantiates Document._Manager and verifies _spawn_doc_server is True.
        """
        nonexisting = os.path.join(tempfile.gettempdir(), f'lazyllm_nonexist_{os.getpid()}')
        assert not os.path.exists(nonexisting), 'Test path should not exist'
        milvus_conf = {'type': 'milvus', 'kwargs': {'uri': 'http://localhost:19530'}}

        mgr = Document._Manager(
            dataset_path=nonexisting, embed=None, manager=False, server=False,
            name='__default__', launcher=None, store_conf=milvus_conf, doc_fields=None,
        )
        # Auto-upgrade should have set _spawn_doc_server = True
        assert mgr._spawn_doc_server is True, (
            'Persistent store + non-existing dir should auto-upgrade to DocServer'
        )
        mgr.stop()

    def test_persistent_store_no_upgrade_for_single_file(self):
        """Regression: auto-upgrade must NOT trigger when dataset_path is an existing file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'test')
            single_file = f.name
        try:
            milvus_conf = {'type': 'milvus', 'kwargs': {'uri': 'http://localhost:19530'}}
            mgr = Document._Manager(
                dataset_path=single_file, embed=None, manager=False, server=False,
                name='__default__', launcher=None, store_conf=milvus_conf, doc_fields=None,
            )
            # Single file should NOT trigger auto-upgrade
            assert mgr._spawn_doc_server is False, (
                'Persistent store + single file should NOT auto-upgrade to DocServer'
            )
            mgr.stop()
        finally:
            os.unlink(single_file)

    def test_persistent_store_existing_dir_auto_upgrade(self):
        """Regression: auto-upgrade must trigger for existing directory + persistent store."""
        dataset_path = self._build_dataset()
        milvus_conf = {'type': 'milvus', 'kwargs': {'uri': 'http://localhost:19530'}}

        mgr = Document._Manager(
            dataset_path=dataset_path, embed=None, manager=False, server=False,
            name='__default__', launcher=None, store_conf=milvus_conf, doc_fields=None,
        )
        assert mgr._spawn_doc_server is True, (
            'Persistent store + existing dir should auto-upgrade to DocServer'
        )
        mgr.stop()

    def test_map_store_no_auto_upgrade(self):
        """Map store should never trigger auto-upgrade regardless of dataset_path."""
        dataset_path = self._build_dataset()
        map_conf = {'type': 'map'}

        mgr = Document._Manager(
            dataset_path=dataset_path, embed=None, manager=False, server=False,
            name='__default__', launcher=None, store_conf=map_conf, doc_fields=None,
        )
        assert mgr._spawn_doc_server is False, 'Map store should never auto-upgrade'
        mgr.stop()

    def test_create_kb_group_before_start_registers_in_doc_server(self):
        """Regression: create_kb_group() before start() must still register KB in DocServer DB.

        Lifecycle: Document(manager=True) → create_kb_group('foo') → start()
        After start(), 'foo' must appear in DocServer's active KB list for scan.
        """
        dataset_path = self._build_dataset()
        doc = Document(dataset_path, manager=True)
        try:
            doc.create_kb_group(name='pre_start_group')
            doc.start()

            # Access the inner DocServer._Impl manager to check DB state
            doc_server = doc._manager._manager
            raw_impl = getattr(doc_server, '_raw_impl', doc_server)
            impl = getattr(raw_impl, '_impl', raw_impl)
            if hasattr(impl, '_lazy_init'):
                impl._lazy_init()
            inner_manager = impl._manager
            pairs = inner_manager._list_active_kb_algo_pairs()
            kb_ids = {kb_id for kb_id, _ in pairs}
            assert 'pre_start_group' in kb_ids, (
                f'KB "pre_start_group" created before start() must be in active pairs, got: {kb_ids}'
            )
        finally:
            doc.stop()

    def test_pre_start_group_parser_algo_registered(self):
        """P1 Regression: create_kb_group before start() must register algorithm with parser.

        Lifecycle: Document(manager=True) → create_kb_group('foo') → start()
        After start(), the DocImpl for 'foo' must have been initialized (i.e. _lazy_init
        called) so that register_algorithm was invoked on the parser.  Without this,
        scan will report 'Invalid algo_id foo' because the parser does not know 'foo'.
        """
        dataset_path = self._build_dataset()
        doc = Document(dataset_path, manager=True)
        try:
            doc.create_kb_group(name='foo')
            doc.start()

            from lazyllm import ServerModule
            kbs = (doc._manager._kbs._impl._m
                   if isinstance(doc._manager._kbs, ServerModule)
                   else doc._manager._kbs)
            # The default group must be initialized
            assert kbs[doc._curr_group]._lazy_init.flag, (
                'Default DocImpl must be initialized after start()'
            )
            # The pre-start group must also be initialized (algorithm registered)
            assert kbs['foo']._lazy_init.flag, (
                'DocImpl for pre-start group "foo" must be initialized after start() '
                'so its algorithm is registered with the parser (not just in the DB)'
            )
        finally:
            doc.stop()

    def test_scan_isolation_between_document_instances(self):
        """P2 Regression: scan must only process KBs owned by the current Document instance.

        DocServer._Impl._sync_dataset filters kb_algo_pairs by _owned_kbs.
        This test creates a DocServer._Impl directly (bypassing ServerModule) and
        verifies that _sync_dataset only processes KBs registered via ensure_kb_registered.
        """
        import os
        import tempfile
        from lazyllm.tools.rag.doc_service.doc_server import DocServer as _DocServer
        from lazyllm.tools.rag.doc_service.parser_client import ParserClient
        from lazyllm.tools.rag.utils import BaseResponse

        dataset_path = self._build_dataset()

        # Create a DocServer._Impl directly (no ServerModule indirection) with mocked parser
        original_health = ParserClient.health
        ParserClient.health = lambda self: BaseResponse(code=200, msg='success', data={'ok': True})
        try:
            with tempfile.TemporaryDirectory(prefix='lazyllm_scan_iso_') as db_dir:
                db_config = {
                    'db_type': 'sqlite', 'user': None, 'password': None,
                    'host': None, 'port': None,
                    'db_name': os.path.join(db_dir, 'scan_iso.db'),
                }
                impl = _DocServer._Impl(
                    storage_dir=dataset_path,
                    db_config=db_config,
                    parser_url='http://mock-parser.test',
                    enable_scan=False,  # don't start scan thread
                )
                impl._lazy_init()

                # Register KBs from "instance 1"
                impl.ensure_kb_registered('kb_inst1_a')
                impl.ensure_kb_registered('kb_inst1_b')

                # Simulate KBs from "instance 2" by inserting directly into DB
                impl._manager._ensure_kb('kb_inst2_x', display_name='kb_inst2_x')
                impl._manager._ensure_kb_algorithm('kb_inst2_x', 'kb_inst2_x')

                # All KBs exist in DB
                all_pairs = impl._manager._list_active_kb_algo_pairs()
                all_kb_ids = {kb for kb, _ in all_pairs}
                assert 'kb_inst1_a' in all_kb_ids
                assert 'kb_inst1_b' in all_kb_ids
                assert 'kb_inst2_x' in all_kb_ids

                # _owned_kbs should only contain instance 1's KBs
                assert impl._owned_kbs == {'kb_inst1_a', 'kb_inst1_b'}, (
                    f'_owned_kbs should only contain registered KBs, got: {impl._owned_kbs}'
                )

                # Track which KBs _sync_dataset_for_kb is called for
                synced_kbs = []
                original_sync_for_kb = impl._sync_dataset_for_kb

                def tracking_sync(kb_id, algo_id, disk_files, disk_set):
                    synced_kbs.append(kb_id)

                impl._sync_dataset_for_kb = tracking_sync
                impl._sync_dataset()

                # Only instance 1's KBs should be synced
                assert 'kb_inst1_a' in synced_kbs, 'owned KB kb_inst1_a must be synced'
                assert 'kb_inst1_b' in synced_kbs, 'owned KB kb_inst1_b must be synced'
                assert 'kb_inst2_x' not in synced_kbs, (
                    'non-owned KB kb_inst2_x must NOT be synced by this instance'
                )
        finally:
            ParserClient.health = original_health

    def test_first_scan_deferred_until_enable_scanning(self):
        """P1 Regression: the very first scan must NOT run during DocServer._lazy_init.

        DocServer._lazy_init() used to eagerly call _sync_dataset(), which ran before
        _owned_kbs was populated and before parser algorithms were registered.  Now
        scanning is deferred until enable_scanning() is called explicitly.
        """
        import os
        import tempfile
        from lazyllm.tools.rag.doc_service.doc_server import DocServer as _DocServer
        from lazyllm.tools.rag.doc_service.parser_client import ParserClient
        from lazyllm.tools.rag.utils import BaseResponse

        dataset_path = self._build_dataset()

        original_health = ParserClient.health
        ParserClient.health = lambda self: BaseResponse(code=200, msg='success', data={'ok': True})
        try:
            with tempfile.TemporaryDirectory(prefix='lazyllm_deferred_scan_') as db_dir:
                db_config = {
                    'db_type': 'sqlite', 'user': None, 'password': None,
                    'host': None, 'port': None,
                    'db_name': os.path.join(db_dir, 'deferred.db'),
                }
                # Track all _sync_dataset_for_kb calls
                sync_log = []
                orig_sync_for_kb = _DocServer._Impl._sync_dataset_for_kb

                def tracking_sync_for_kb(self_inner, kb_id, algo_id, disk_files, disk_set):
                    sync_log.append(kb_id)

                _DocServer._Impl._sync_dataset_for_kb = tracking_sync_for_kb
                try:
                    impl = _DocServer._Impl(
                        storage_dir=dataset_path,
                        db_config=db_config,
                        parser_url='http://mock-parser.test',
                        enable_scan=True,  # scanning is ENABLED but deferred
                    )

                    # _lazy_init must NOT trigger _sync_dataset
                    impl._lazy_init()
                    assert sync_log == [], (
                        f'_lazy_init must NOT call _sync_dataset, but synced: {sync_log}'
                    )

                    # Simulate the Document._Manager startup sequence:
                    # 1. Register KB
                    impl.ensure_kb_registered('my_group')
                    assert sync_log == [], (
                        f'ensure_kb_registered must NOT trigger scan, but synced: {sync_log}'
                    )

                    # 2. Now explicitly enable scanning
                    impl.enable_scanning()
                    assert sync_log == ['my_group'], (
                        f'enable_scanning must trigger first scan for owned KBs only, '
                        f'got: {sync_log}'
                    )
                finally:
                    _DocServer._Impl._sync_dataset_for_kb = orig_sync_for_kb
        finally:
            ParserClient.health = original_health
