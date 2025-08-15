import os
import tempfile
import unittest
import pytest

from unittest.mock import MagicMock

from lazyllm.tools.rag.store.document_store import _DocumentStore
from lazyllm.tools.rag.store import MapStore, MilvusStore, BUILDIN_GLOBAL_META_DESC, HybridStore
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.rag.doc_node import DocNode, QADocNode, ImageDocNode

node1 = DocNode(uid="1", text="text1", group="group1", parent=None,
                global_metadata={RAG_KB_ID: "kb1", RAG_DOC_ID: "doc1", "tags": ["tag1"]})
node2 = DocNode(uid="2", text="text2", group="group1", parent=None,
                global_metadata={RAG_KB_ID: "kb2", RAG_DOC_ID: "doc2", "tags": ["tag2"]})
node3 = DocNode(uid="3", text="text3", group="group2", parent=node1,
                global_metadata={RAG_KB_ID: "kb3", RAG_DOC_ID: "doc3", "tags": ["tag3"]})
qa_node1 = QADocNode(uid="4", query="query1", answer="answer1", group="qa", parent=node1,
                     global_metadata={RAG_KB_ID: "kb1", RAG_DOC_ID: "doc3", "tags": ["tag4"]})
image_node1 = ImageDocNode(uid="5", image_path="image1.png", group="image", parent=node1,
                           global_metadata={RAG_KB_ID: "kb1", RAG_DOC_ID: "doc4", "tags": ["tag5"]})


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestStoreWithMapAndMilvus(unittest.TestCase):
    def setUp(self):
        fd, self.store_dir = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.mock_embed = {
            'vec_dense': MagicMock(return_value=[1.0, 2.0, 3.0]),
            'vec_sparse': MagicMock(return_value={0: 1.0, 1: 2.0, 2: 3.0}),
        }
        self.index_kwargs = [
            {
                'embed_key': 'vec_dense',
                'index_type': 'FLAT',
                'metric_type': 'COSINE',
                'params': {
                    'nlist': 128,
                }
            },
            {
                'embed_key': 'vec_sparse',
                'index_type': 'SPARSE_INVERTED_INDEX',
                'metric_type': 'IP',
                'params': {
                    'nlist': 128,
                }
            }
        ]
        self.embed_dims = {"vec_dense": 3}
        self.embed_datatypes = {"vec_dense": DataType.FLOAT_VECTOR, "vec_sparse": DataType.SPARSE_FLOAT_VECTOR}
        self.group_embed_keys = {
            "group1": {"vec_dense", "vec_sparse"},
            "group2": {"vec_dense", "vec_sparse"},
            "qa": {"vec_dense", "vec_sparse"},
            "image": {}
        }
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.document_store = _DocumentStore(algo_name="__default__",
                                             store=HybridStore(MapStore(),
                                                               MilvusStore(uri=self.store_dir,
                                                                           index_kwargs=self.index_kwargs)),
                                             group_embed_keys=self.group_embed_keys,
                                             embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                                             embed=self.mock_embed,
                                             global_metadata_desc=self.global_metadata_desc)
        self.document_store.activate_group(["group1", "group2", "qa", "image"])
        self.document_store.update_nodes([node1, node2, node3, qa_node1, image_node1])

    def tearDown(self):
        os.remove(self.store_dir)

    def test_initialization(self):
        self.assertEqual(set(self.document_store.activated_groups()),
                         set(["group1", "group2", "qa", "image"]))

    def test_get_nodes_by_group(self):
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid, node2.uid]))
        nodes = self.document_store.get_nodes(group="group2")
        self.assertEqual(set([node.uid for node in nodes]), set([node3.uid]))
        self.assertEqual(nodes[0].parent, node1.uid)
        nodes = self.document_store.get_nodes(group="qa")
        self.assertEqual(set([node.uid for node in nodes]), set([qa_node1.uid]))
        self.assertEqual(isinstance(nodes[0], QADocNode), True)
        nodes = self.document_store.get_nodes(group="image")
        self.assertEqual(set([node.uid for node in nodes]), set([image_node1.uid]))
        self.assertEqual(isinstance(nodes[0], ImageDocNode), True)

    def test_get_nodes_by_doc_id(self):
        nodes = self.document_store.get_nodes(group="group1", doc_ids=[node1.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid]))
        nodes = self.document_store.get_nodes(group="group1", doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        nodes = self.document_store.get_nodes(group="group2", doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(len(nodes), 0)
        nodes = self.document_store.get_nodes(group="group3", doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(len(nodes), 0)

    def test_get_nodes_by_kb_id(self):
        nodes = self.document_store.get_nodes(group="group1", kb_id=node1.global_metadata.get(RAG_KB_ID))
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid]))
        nodes = self.document_store.get_nodes(group="group2", kb_id=node3.global_metadata.get(RAG_KB_ID))
        self.assertEqual(set([node.uid for node in nodes]), set([node3.uid]))
        nodes = self.document_store.get_nodes(group="group3", kb_id=node3.global_metadata.get(RAG_KB_ID))
        self.assertEqual(len(nodes), 0)

    def test_get_nodes_by_uids(self):
        nodes = self.document_store.get_nodes(group="group1", uids=[node1.uid])
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid]))
        nodes = self.document_store.get_nodes(group="group1", uids=[node2.uid])
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        nodes = self.document_store.get_nodes(group="group2", uids=[node3.uid])
        self.assertEqual(set([node.uid for node in nodes]), set([node3.uid]))
        nodes = self.document_store.get_nodes(group="group3", uids=[node3.uid])
        self.assertEqual(len(nodes), 0)

    def test_remove_nodes_by_uids(self):
        self.document_store.remove_nodes(group="group1", uids=[node1.uid])
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        self.document_store.remove_nodes(group="group1", uids=[node2.uid])
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(len(nodes), 0)
        self.document_store.remove_nodes(group="group2", uids=[node3.uid])
        nodes = self.document_store.get_nodes(group="group2")
        self.assertEqual(len(nodes), 0)

    def test_remove_nodes_by_doc_id(self):
        self.document_store.remove_nodes(group="group1", doc_ids=[node1.global_metadata.get(RAG_DOC_ID)])
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        self.document_store.remove_nodes(group="group1", doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(len(nodes), 0)
        self.document_store.remove_nodes(group="group2", doc_ids=[node3.global_metadata.get(RAG_DOC_ID)])
        nodes = self.document_store.get_nodes(group="group2")
        self.assertEqual(len(nodes), 0)

    def test_remove_nodes_by_kb_id(self):
        self.document_store.remove_nodes(group="group1", kb_id=node1.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        self.document_store.remove_nodes(group="group1", kb_id=node2.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(len(nodes), 0)
        self.document_store.remove_nodes(group="group2", kb_id=node3.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group="group2")
        self.assertEqual(len(nodes), 0)
        self.document_store.update_nodes([node1, node2, node3])
        self.document_store.remove_nodes(kb_id=node1.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group="group1")
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        nodes = self.document_store.get_nodes(group="group2")
        self.assertEqual(len(nodes), 0)

    def test_update_doc_meta(self):
        self.document_store.update_doc_meta(node1.global_metadata.get(RAG_DOC_ID), {"tags": ["updated_tag"]})
        nodes = self.document_store.get_nodes(kb_id=node1.global_metadata.get(RAG_KB_ID))
        self.assertEqual(len(nodes), 4)
        for node in nodes:
            self.assertEqual(node.global_metadata.get("tags"), ["updated_tag"])

    def test_query_without_filters(self):
        nodes = self.document_store.query(query="text1", group_name="group1", embed_keys=["vec_dense"], topk=2)
        self.assertEqual(len(nodes), 2)
        nodes = self.document_store.query(query="text1", group_name="qa", embed_keys=["vec_dense"], topk=2)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uid, qa_node1.uid)

    def test_query_with_filters(self):
        nodes = self.document_store.query(query="text1", group_name="group1", embed_keys=["vec_dense"],
                                          topk=2, filters={RAG_DOC_ID: ["doc1"]})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uid, node1.uid)
        nodes = self.document_store.query(query="text1", group_name="group1", embed_keys=["vec_dense"],
                                          topk=2, filters={RAG_DOC_ID: ["doc2"]})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uid, node2.uid)
