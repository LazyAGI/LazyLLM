import os
import tempfile
import unittest
import pytest
import json

from unittest.mock import MagicMock

from lazyllm.tools.rag.store.document_store import _DocumentStore
from lazyllm.tools.rag.store import MapStore, MilvusStore, BUILDIN_GLOBAL_META_DESC, HybridStore
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.rag.doc_node import DocNode, QADocNode, ImageDocNode, JsonDocNode, RichDocNode, MetadataMode

node1 = DocNode(uid='1', text='text1', group='group1', parent=None,
                global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc1', 'tags': ['tag1']})
node2 = DocNode(uid='2', text='text2', group='group1', parent=None,
                global_metadata={RAG_KB_ID: 'kb2', RAG_DOC_ID: 'doc2', 'tags': ['tag2']})
node3 = DocNode(uid='3', text='text3', group='group2', parent=node1,
                global_metadata={RAG_KB_ID: 'kb3', RAG_DOC_ID: 'doc3', 'tags': ['tag3']})
qa_node1 = QADocNode(uid='4', query='query1', answer='answer1', group='qa', parent=node1,
                     global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc3', 'tags': ['tag4']})
image_node1 = ImageDocNode(uid='5', image_path='image1.png', group='image', parent=node1,
                           global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc4', 'tags': ['tag5']})
json_node1 = JsonDocNode(uid='6', content={'key': 'value', 'nested': {'a': 1, 'b': 2}},
                         group='json', parent=node1, formatter_str='[key,nested]',
                         metadata={'custom_meta': 'test'},
                         global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc5', 'tags': ['tag6']})
rich_node1 = RichDocNode(uid='7', nodes=[node1, node2], group='rich', parent=None,
                         global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc6', 'tags': ['tag7']})


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestStoreWithMapAndMilvus(unittest.TestCase):
    def setUp(self):
        fd, self.store_dir = tempfile.mkstemp(suffix='.db')
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
        self.embed_dims = {'vec_dense': 3}
        self.embed_datatypes = {'vec_dense': DataType.FLOAT_VECTOR, 'vec_sparse': DataType.SPARSE_FLOAT_VECTOR}
        self.group_embed_keys = {
            'group1': {'vec_dense', 'vec_sparse'},
            'group2': {'vec_dense', 'vec_sparse'},
            'qa': {'vec_dense', 'vec_sparse'},
            'image': {}
        }
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.document_store = _DocumentStore(algo_name='__default__',
                                             store=HybridStore(MapStore(),
                                                               MilvusStore(uri=self.store_dir,
                                                                           index_kwargs=self.index_kwargs)),
                                             group_embed_keys=self.group_embed_keys,
                                             embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                                             embed=self.mock_embed,
                                             global_metadata_desc=self.global_metadata_desc)
        self.document_store.activate_group(['group1', 'group2', 'qa', 'image', 'json', 'rich'])
        self.document_store.update_nodes([node1, node2, node3, qa_node1, image_node1, json_node1, rich_node1])

    def tearDown(self):
        os.remove(self.store_dir)

    def test_initialization(self):
        self.assertEqual(set(self.document_store.activated_groups()),
                         set(['group1', 'group2', 'qa', 'image', 'json', 'rich']))

    def test_get_nodes_by_group(self):
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid, node2.uid]))
        nodes = self.document_store.get_nodes(group='group2')
        self.assertEqual(set([node.uid for node in nodes]), set([node3.uid]))
        self.assertEqual(nodes[0].parent, node1.uid)
        nodes = self.document_store.get_nodes(group='qa')
        self.assertEqual(set([node.uid for node in nodes]), set([qa_node1.uid]))
        self.assertEqual(isinstance(nodes[0], QADocNode), True)
        nodes = self.document_store.get_nodes(group='image')
        self.assertEqual(set([node.uid for node in nodes]), set([image_node1.uid]))
        self.assertEqual(isinstance(nodes[0], ImageDocNode), True)
        nodes = self.document_store.get_nodes(group='json')
        self.assertEqual(set([node.uid for node in nodes]), set([json_node1.uid]))
        self.assertEqual(isinstance(nodes[0], JsonDocNode), True)
        nodes = self.document_store.get_nodes(group='rich')
        self.assertEqual(set([node.uid for node in nodes]), set([rich_node1.uid]))
        self.assertEqual(isinstance(nodes[0], RichDocNode), True)

    def test_get_nodes_by_doc_id(self):
        nodes = self.document_store.get_nodes(group='group1', doc_ids=[node1.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid]))
        nodes = self.document_store.get_nodes(group='group1', doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        nodes = self.document_store.get_nodes(group='group2', doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(len(nodes), 0)
        nodes = self.document_store.get_nodes(group='group3', doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        self.assertEqual(len(nodes), 0)

    def test_get_nodes_by_kb_id(self):
        nodes = self.document_store.get_nodes(group='group1', kb_id=node1.global_metadata.get(RAG_KB_ID))
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid]))
        nodes = self.document_store.get_nodes(group='group2', kb_id=node3.global_metadata.get(RAG_KB_ID))
        self.assertEqual(set([node.uid for node in nodes]), set([node3.uid]))
        nodes = self.document_store.get_nodes(group='group3', kb_id=node3.global_metadata.get(RAG_KB_ID))
        self.assertEqual(len(nodes), 0)

    def test_get_nodes_by_uids(self):
        nodes = self.document_store.get_nodes(group='group1', uids=[node1.uid])
        self.assertEqual(set([node.uid for node in nodes]), set([node1.uid]))
        nodes = self.document_store.get_nodes(group='group1', uids=[node2.uid])
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        nodes = self.document_store.get_nodes(group='group2', uids=[node3.uid])
        self.assertEqual(set([node.uid for node in nodes]), set([node3.uid]))
        nodes = self.document_store.get_nodes(group='group3', uids=[node3.uid])
        self.assertEqual(len(nodes), 0)

    def test_remove_nodes_by_uids(self):
        self.document_store.remove_nodes(group='group1', uids=[node1.uid])
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        self.document_store.remove_nodes(group='group1', uids=[node2.uid])
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(len(nodes), 0)
        self.document_store.remove_nodes(group='group2', uids=[node3.uid])
        nodes = self.document_store.get_nodes(group='group2')
        self.assertEqual(len(nodes), 0)

    def test_remove_nodes_by_doc_id(self):
        self.document_store.remove_nodes(group='group1', doc_ids=[node1.global_metadata.get(RAG_DOC_ID)])
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        self.document_store.remove_nodes(group='group1', doc_ids=[node2.global_metadata.get(RAG_DOC_ID)])
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(len(nodes), 0)
        self.document_store.remove_nodes(group='group2', doc_ids=[node3.global_metadata.get(RAG_DOC_ID)])
        nodes = self.document_store.get_nodes(group='group2')
        self.assertEqual(len(nodes), 0)

    def test_remove_nodes_by_kb_id(self):
        self.document_store.remove_nodes(group='group1', kb_id=node1.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        self.document_store.remove_nodes(group='group1', kb_id=node2.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(len(nodes), 0)
        self.document_store.remove_nodes(group='group2', kb_id=node3.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group='group2')
        self.assertEqual(len(nodes), 0)
        self.document_store.update_nodes([node1, node2, node3])
        self.document_store.remove_nodes(kb_id=node1.global_metadata.get(RAG_KB_ID))
        nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(set([node.uid for node in nodes]), set([node2.uid]))
        nodes = self.document_store.get_nodes(group='group2')
        self.assertEqual(len(nodes), 0)

    def test_query_without_filters(self):
        nodes = self.document_store.query(query='text1', group_name='group1', embed_keys=['vec_dense'], topk=2)
        self.assertEqual(len(nodes), 2)
        nodes = self.document_store.query(query='text1', group_name='qa', embed_keys=['vec_dense'], topk=2)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uid, qa_node1.uid)

    def test_query_with_filters(self):
        nodes = self.document_store.query(query='text1', group_name='group1', embed_keys=['vec_dense'],
                                          topk=2, filters={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uid, node1.uid)
        nodes = self.document_store.query(query='text1', group_name='group1', embed_keys=['vec_dense'],
                                          topk=2, filters={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].uid, node2.uid)

    def test_get_nodes_pagination_limit(self):
        nodes = self.document_store.get_nodes(limit=2)
        self.assertEqual(len(nodes), 2)

    def test_get_nodes_pagination_offset(self):
        _, total = self.document_store.get_nodes(return_total=True)
        nodes_offset = self.document_store.get_nodes(limit=2, offset=total)
        self.assertEqual(len(nodes_offset), 0)

    def test_get_nodes_pagination_return_total(self):
        nodes, total = self.document_store.get_nodes(limit=2, return_total=True)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(total, 7)  # 2 in group1, 1 in group2, 1 in qa, 1 in image, 1 in json, 1 in rich

    def test_get_nodes_pagination_with_group(self):
        nodes = self.document_store.get_nodes(group='group1', limit=1)
        self.assertEqual(len(nodes), 1)

    def test_get_segments_pagination_limit(self):
        segments = self.document_store.get_segments(limit=2)
        self.assertEqual(len(segments), 2)

    def test_get_segments_pagination_return_total(self):
        segments, total = self.document_store.get_segments(limit=3, return_total=True)
        self.assertEqual(len(segments), 3)
        self.assertEqual(total, 7)  # 2 in group1, 1 in group2, 1 in qa, 1 in image, 1 in json, 1 in rich

    def test_get_segments_pagination_with_doc_id(self):
        segments = self.document_store.get_segments(doc_ids={node1.global_metadata.get(RAG_DOC_ID)}, limit=1)
        self.assertEqual(len(segments), 1)

    def test_json_node_store_and_retrieve(self):
        '''Test that JsonDocNode can be stored and correctly reconstructed from store.'''
        # Retrieve the node from store
        nodes = self.document_store.get_nodes(group='json', uids=[json_node1.uid])
        self.assertEqual(len(nodes), 1)
        retrieved_node = nodes[0]

        # Verify it's a JsonDocNode
        self.assertIsInstance(retrieved_node, JsonDocNode)

        # Verify content is correctly reconstructed
        self.assertEqual(retrieved_node.json_object, json_node1.json_object)
        self.assertEqual(retrieved_node.json_object, {'key': 'value', 'nested': {'a': 1, 'b': 2}})

        # Verify formatter_str is preserved in metadata
        self.assertEqual(retrieved_node.metadata.get('formatter_str'), '[key,nested]')

        # Verify global_metadata is preserved
        self.assertEqual(retrieved_node.global_metadata.get(RAG_DOC_ID), 'doc1')
        self.assertEqual(retrieved_node.global_metadata.get('tags'), ['tag1'])

        # Verify formatter works correctly after reconstruction
        content = retrieved_node.get_content(metadata_mode=MetadataMode.EMBED)
        # The formatter should extract [key, nested] fields
        formatted = json.loads(content)
        self.assertEqual(formatted, ['value', {'a': 1, 'b': 2}])

        # Verify text property returns JSON string
        text = retrieved_node.text
        parsed_text = json.loads(text)
        self.assertEqual(parsed_text, {'key': 'value', 'nested': {'a': 1, 'b': 2}})

    def test_rich_node_store_and_retrieve(self):
        '''Test that RichDocNode can be stored and correctly reconstructed from store.'''
        # Retrieve the node from store
        nodes = self.document_store.get_nodes(group='rich', uids=[rich_node1.uid])
        self.assertEqual(len(nodes), 1)
        retrieved_node = nodes[0]

        # Verify it's a RichDocNode
        self.assertIsInstance(retrieved_node, RichDocNode)

        # Verify sub-nodes are correctly reconstructed
        self.assertEqual(len(retrieved_node.nodes), 2)
        self.assertEqual(retrieved_node.nodes[0].text, 'text1')
        self.assertEqual(retrieved_node.nodes[1].text, 'text2')

        # Verify content is correctly reconstructed
        doc_nodes = self.document_store.get_nodes(group='group1')
        self.assertEqual(sorted(retrieved_node.content), sorted([doc_nodes[0].text, doc_nodes[1].text]))
        self.assertEqual(retrieved_node.text, '\n'.join(sorted([doc_nodes[0].text, doc_nodes[1].text])))

        # Verify metadata of sub-nodes (partially)
        self.assertEqual(retrieved_node.nodes[0].global_metadata.get(RAG_DOC_ID), 'doc1')
        self.assertEqual(retrieved_node.nodes[1].global_metadata.get(RAG_DOC_ID), 'doc2')

        # Verify global_metadata of RichDocNode itself
        self.assertEqual(retrieved_node.global_metadata.get(RAG_DOC_ID), 'doc6')
        self.assertEqual(retrieved_node.global_metadata.get('tags'), ['tag7'])
