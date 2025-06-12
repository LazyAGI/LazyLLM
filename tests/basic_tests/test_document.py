import lazyllm
from lazyllm.tools.rag.doc_impl import DocImpl
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.store_base import LAZY_ROOT_NAME
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_PATH, RAG_DOC_ID
from lazyllm.tools.rag import Document, Retriever, TransformArgs, AdaptiveTransform, TempDocRetriever
from lazyllm.tools.rag.doc_manager import DocManager
from lazyllm.tools.rag.utils import DocListManager
from lazyllm.launcher import cleanup
from lazyllm import config
from unittest.mock import MagicMock
import unittest
import httpx
import os
import shutil
import io
import re
import json
import time
import tempfile


class TestDocImpl(unittest.TestCase):

    def setUp(self):
        self.mock_embed = MagicMock()
        self.mock_directory_reader = MagicMock()
        # use temporary file as only existing files can be added to DocImpl
        self.tmp_file_a = tempfile.NamedTemporaryFile()
        self.tmp_file_b = tempfile.NamedTemporaryFile()
        mock_node = DocNode(group=LAZY_ROOT_NAME, text="dummy text")
        mock_node._global_metadata = {RAG_DOC_PATH: self.tmp_file_a.name}
        self.mock_directory_reader.load_data.return_value = ([mock_node], [])

        self.doc_impl = DocImpl(embed=self.mock_embed, doc_files=[self.tmp_file_a.name])
        self.doc_impl._reader = self.mock_directory_reader

    def tearDown(self):
        self.tmp_file_a.close()
        self.tmp_file_b.close()

    def test_create_node_group_default(self):
        self.doc_impl._create_builtin_node_group('MyChunk', transform=lambda x: ','.split(x))
        self.doc_impl._lazy_init()
        assert "MyChunk" in self.doc_impl.node_groups
        assert "CoarseChunk" in self.doc_impl.node_groups
        assert "MediumChunk" in self.doc_impl.node_groups
        assert "FineChunk" in self.doc_impl.node_groups

    def test_create_node_group(self):
        self.doc_impl._lazy_init.flag.reset()
        self.doc_impl.create_node_group(
            name="CustomChunk",
            transform=SentenceSplitter,
            chunk_size=512,
            chunk_overlap=50,
        )
        assert "CustomChunk" in self.doc_impl.node_groups
        node_group = self.doc_impl.node_groups["CustomChunk"]
        assert node_group["transform"].f == SentenceSplitter
        assert node_group["transform"].kwargs["chunk_size"] == 512
        assert node_group["transform"]["kwargs"]["chunk_overlap"] == 50

    def test_retrieve(self):
        self.mock_embed.return_value = "[0.1, 0.2, 0.3]"
        self.doc_impl.activate_group(Document.FineChunk, [])
        result = self.doc_impl.retrieve(
            query="test query",
            group_name="FineChunk",
            similarity="bm25",
            similarity_cut_off=-100,
            index='default',
            topk=1,
            similarity_kws={},
        )
        node = result[0]
        assert node.text == "dummy text"

    def test_add_files(self):
        assert self.doc_impl.store is None
        self.doc_impl._lazy_init()
        assert len(self.doc_impl.store.get_nodes(LAZY_ROOT_NAME)) == 1
        new_doc = DocNode(text="new dummy text", group=LAZY_ROOT_NAME)
        new_doc._global_metadata = {RAG_DOC_PATH: self.tmp_file_b.name}
        self.mock_directory_reader.load_data.return_value = ([new_doc], [])
        self.doc_impl._add_doc_to_store([self.tmp_file_b.name])
        assert len(self.doc_impl.store.get_nodes(LAZY_ROOT_NAME)) == 2

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
            dict(f=SentenceSplitter, pattern='*.txt', kwargs=dict(chunk_size=512, chunk_overlap=50)),
            TransformArgs(f=SentenceSplitter, pattern=None, kwargs=dict(chunk_size=256, chunk_overlap=25))]))
        doc = Document('rag_master')
        doc._impl._lazy_init()
        retriever = Retriever(doc, 'AdaptiveChunk1', similarity='bm25', topk=2)
        retriever('什么是道')
        retriever = Retriever(doc, 'AdaptiveChunk2', similarity='bm25', topk=2)
        retriever('什么是道')

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


class TestTempRetriever():
    def test_temp_retriever(self):
        r = TempDocRetriever()(os.path.join(config['data_path'], 'rag_master/default/__data/sources/大学.txt'), '大学')
        assert len(r) > 0 and isinstance(r[0], DocNode)

        r = TempDocRetriever(output_format='content')('rag_master/default/__data/sources/大学.txt', '大学')
        assert len(r) > 0 and isinstance(r[0], str)

        ret = TempDocRetriever(output_format='dict')
        ret.create_node_group('block', transform=lambda x: x.split('\n'))
        ret.add_subretriever(Document.CoarseChunk, topk=1)
        ret.add_subretriever('block', topk=3)
        r = ret(['rag_master/default/__data/sources/大学.txt', 'rag_master/default/__data/sources/论语.txt'], '大学')
        assert len(r) == 4 and isinstance(r[0], dict)
        r = ret(['rag_master/default/__data/sources/大学.txt', 'rag_master/default/__data/sources/论语.txt'], '大学')
        assert len(r) == 4 and isinstance(r[0], dict)
        r = ret(['rag_master/default/__data/sources/论语.txt', 'rag_master/default/__data/sources/大学.txt'], '大学')
        assert len(r) == 4 and isinstance(r[0], dict)


class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(config['home'], 'rag_for_document_ut'))
        self.rag_dir = os.path.join(self.root_dir, 'rag_master')
        os.makedirs(self.rag_dir, exist_ok=True)

    def __del__(self):
        shutil.rmtree(self.root_dir)

class TestDocumentServer(unittest.TestCase):
    def setUp(self):
        self.dir = TmpDir()
        self.dlm = DocListManager(path=self.dir.rag_dir, name=None, enable_path_monitoring=False)

        self.doc_impl = DocImpl(embed=MagicMock(), dlm=self.dlm)
        self.doc_impl._lazy_init()

        doc_manager = DocManager(self.dlm)
        self.server = lazyllm.ServerModule(doc_manager)

        self.server.start()

        url_pattern = r'(http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+)'
        self.doc_server_addr = re.findall(url_pattern, self.server._url)[0]

    def test_delete_files_in_store(self):
        files = [('files', ('test1.txt', io.BytesIO(b"John's house is in Beijing"), 'text/palin')),
                 ('files', ('test2.txt', io.BytesIO(b"John's house is in Shanghai"), 'text/plain'))]
        metadatas = [{"comment": "comment1"}, {"signature": "signature2"}]
        params = dict(override='true', metadatas=json.dumps(metadatas))

        url = f'{self.doc_server_addr}/upload_files'
        response = httpx.post(url, params=params, files=files, timeout=10)
        assert response.status_code == 200 and response.json().get('code') == 200, response.json()
        ids = response.json().get('data')[0]
        lazyllm.LOG.info(f'debug!!! ids -> {ids}')
        assert len(ids) == 2

        time.sleep(20)  # waiting for worker thread to update newly uploaded files

        # make sure that ids are written into the store
        nodes = self.doc_impl.store.get_nodes(LAZY_ROOT_NAME)
        for node in nodes:
            if node.global_metadata[RAG_DOC_PATH].endswith('test1.txt'):
                test1_docid = node.global_metadata[RAG_DOC_ID]
            elif node.global_metadata[RAG_DOC_PATH].endswith('test2.txt'):
                test2_docid = node.global_metadata[RAG_DOC_ID]
        assert test1_docid and test2_docid
        assert set([test1_docid, test2_docid]) == set(ids)

        url = f'{self.doc_server_addr}/delete_files'
        response = httpx.post(url, json=dict(file_ids=[test1_docid]))
        assert response.status_code == 200 and response.json().get('code') == 200

        time.sleep(20)  # waiting for worker thread to delete files

        nodes = self.doc_impl.store.get_nodes(LAZY_ROOT_NAME)
        assert len(nodes) == 1
        assert nodes[0].global_metadata[RAG_DOC_ID] == test2_docid
        cur_meta_dict = nodes[0].global_metadata

        url = f'{self.doc_server_addr}/add_metadata'
        response = httpx.post(url, json=dict(doc_ids=[test2_docid], kv_pair={"title": "title2"}))
        assert response.status_code == 200 and response.json().get('code') == 200
        time.sleep(20)
        assert cur_meta_dict["title"] == "title2"

        response = httpx.post(url, json=dict(doc_ids=[test2_docid], kv_pair={"title": "TITLE2"}))
        assert response.status_code == 200 and response.json().get('code') == 200
        time.sleep(20)
        assert cur_meta_dict["title"] == ["title2", "TITLE2"]

        url = f'{self.doc_server_addr}/delete_metadata_item'
        response = httpx.post(url, json=dict(doc_ids=[test2_docid], keys=["signature"]))
        assert response.status_code == 200 and response.json().get('code') == 200
        time.sleep(20)
        assert "signature" not in cur_meta_dict

        response = httpx.post(url, json=dict(doc_ids=[test2_docid], kv_pair={"title": "TITLE2"}))
        assert response.status_code == 200 and response.json().get('code') == 200
        time.sleep(20)
        assert cur_meta_dict["title"] == ["title2"]

        url = f'{self.doc_server_addr}/update_or_create_metadata_keys'
        response = httpx.post(url, json=dict(doc_ids=[test2_docid], kv_pair={"signature": "signature2"}))
        assert response.status_code == 200 and response.json().get('code') == 200
        time.sleep(20)
        assert cur_meta_dict["signature"] == "signature2"

        url = f'{self.doc_server_addr}/reset_metadata'
        response = httpx.post(url, json=dict(doc_ids=[test2_docid],
                                             new_meta={"author": "author2", "signature": "signature_new"}))
        assert response.status_code == 200 and response.json().get('code') == 200
        time.sleep(20)
        assert cur_meta_dict["signature"] == "signature_new" and cur_meta_dict["author"] == "author2"

        url = f'{self.doc_server_addr}/query_metadata'
        response = httpx.post(url, json=dict(doc_id=test2_docid))

        # make sure that only one file is left
        response = httpx.get(f'{self.doc_server_addr}/list_files')
        assert response.status_code == 200 and len(response.json().get('data')) == 1

    def tearDown(self):
        # Must clean up the server as all uploaded files will be deleted as they are in tmp dir
        self.dlm.release()

if __name__ == "__main__":
    unittest.main()
