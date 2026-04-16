from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.store import EMBED_DEFAULT_KEY
from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag import (Document, Retriever, TempDocRetriever, ContextRetriever,
                               WeightedRetriever, PriorityRetriever)
from lazyllm.launcher import cleanup
from lazyllm import config
from unittest.mock import MagicMock
import os

class TestRetriever(object):
    @classmethod
    def tearDownClass(cls):
        cleanup()

    def test_per_doc_embed_keys_align_after_lazy_init(self):
        mock_embed = MagicMock(return_value=[0.4, 0.5, 0.6])
        doc1 = Document('rag_master', embed=mock_embed)
        doc2 = Document('rag_master', embed=mock_embed)
        doc1.create_node_group(
            'local_group',
            parent=Document.CoarseChunk,
            transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=128, chunk_overlap=16)),
        )
        retriever1 = Retriever([doc1, doc2], 'local_group', similarity='cosine', topk=2)
        assert retriever1._per_doc_embed_keys is True
        assert len(retriever1._docs) == 2
        assert retriever1._embed_keys == [[EMBED_DEFAULT_KEY], [EMBED_DEFAULT_KEY]]

        retriever1._lazy_init()
        assert len(retriever1._docs) == 1
        assert retriever1._embed_keys == [[EMBED_DEFAULT_KEY]]

        retriever2 = Retriever(doc1, 'local_group', topk=2, embed_keys=[EMBED_DEFAULT_KEY])
        assert retriever2._similarity == 'cosine'
        assert retriever2._per_doc_embed_keys is False
        assert retriever2._embed_keys == [EMBED_DEFAULT_KEY]

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

    def test_context_retriever(self):
        ctx1 = '大学之道，在明明德，\n在亲民，在止于至善。\n知止而后有定，定而后能静，静而后能安。'
        ctx2 = '子曰：学而时习之，不亦说乎？\n有朋自远方来，不亦乐乎？'

        r = ContextRetriever()(ctx1, '大学')
        assert len(r) > 0 and isinstance(r[0], DocNode)
        r = ContextRetriever(output_format='content')([ctx1, ctx2], '大学')
        assert len(r) > 0 and isinstance(r[0], str)

        ret = ContextRetriever(output_format='dict')
        ret.create_node_group('block', transform=lambda x: x.split('\n'))
        ret.add_subretriever(Document.CoarseChunk, topk=1)
        ret.add_subretriever('block', topk=3)
        r = ret([ctx1, ctx2], '大学')
        assert len(r) == 4 and isinstance(r[0], dict)


class TestCompositeRetriever(object):
    @classmethod
    def tearDownClass(cls):
        cleanup()

    def test_weighted_retriever(self):
        doc = Document('rag_master')
        doc.create_node_group('chunk1', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        with WeightedRetriever(topk=3) as w:
            w.retriever1 = Retriever(doc, 'chunk1', similarity='bm25', topk=3, weight=1)
            w.retriever2 = Retriever(doc, Document.CoarseChunk, similarity='bm25', topk=3, weight=1)
            w.retriever3 = Retriever(doc, Document.FineChunk, similarity='bm25', topk=3, weight=1)
        w.start()
        result = w('什么是道')
        assert len(result) == 3
        assert set(r.group for r in result) == set(['chunk1', Document.CoarseChunk.name, Document.FineChunk.name])

        result = w('什么是道', weights=[2, 0, 1])
        assert len(result) == 3
        assert set(r.group for r in result) == set(['chunk1', Document.FineChunk.name])
        assert len([r.group for r in result if r.group == 'chunk1']) == 2

        result = w('什么是道', weights=[4, 2, 2], topk=7)
        assert len(result) == 7
        assert len([r.group for r in result if r.group == 'chunk1']) == 3

        result = w('什么是道', weights=[4, 4, 4], topk=12)
        assert len(result) == 9

    def test_weighted_retriever_with_temp_retriever(self):
        # TODO: support temp retrievers in weighted retriever
        pass

    def test_priority_retriever(self):
        doc = Document('rag_master')
        doc.create_node_group('chunk1', parent=Document.CoarseChunk,
                              transform=dict(f=SentenceSplitter, kwargs=dict(chunk_size=256, chunk_overlap=25)))
        with PriorityRetriever(topk=3) as w:
            w.retriever1 = Retriever(doc, 'chunk1', similarity='bm25', topk=3, priority=Retriever.Priority.high)
            w.retriever2 = Retriever(doc, Document.CoarseChunk, similarity='bm25', topk=3, priority='low')
            w.retriever3 = Retriever(doc, Document.FineChunk, similarity='bm25', topk=3)
        w.start()

        result = w('什么是道')
        assert len(result) == 3
        assert len([r.group for r in result if r.group == 'chunk1']) == 3

        result = w('什么是道', priorities=[Retriever.Priority.ignore, 'high', 'low'])
        assert len(result) == 3
        assert len([r.group for r in result if r.group == Document.CoarseChunk.name]) == 3

        result = w('什么是道', priorities=['ignore', Retriever.Priority.high, 'low'], topk=7)
        assert len(result) == 6
        assert len([r.group for r in result if r.group == Document.CoarseChunk.name]) == 3
        assert len([r.group for r in result if r.group == Document.FineChunk.name]) == 3

        result = w('什么是道', topk=4)
        assert len(result) == 6
        assert len([r.group for r in result if r.group == 'chunk1']) == 3
        assert len([r.group for r in result if r.group == Document.FineChunk.name]) == 3

    def test_priority_retriever_with_temp_retriever(self):
        # TODO: support temp retrievers in priority retriever
        pass
