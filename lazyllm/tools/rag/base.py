import copy
import json
import hashlib
import importlib

from lazyllm import ModuleBase
from lazyllm.common import LazyLlmRequest
from lazyllm.thirdparty import llama_index
import lazyllm


register_flag = lazyllm.once_flag()

class DocImpl(ModuleBase):
    registered_similarity = dict()
    parser_rule_dict = dict()

    @classmethod
    def register(cls):
        cls.register_parser(name='Hierarchy', transform='HierarchicalNodeParser', chunk_sizes=[1024, 512, 128])
        cls.register_parser(name='CoarseChunk', transform='get_deeper_nodes', depth=0, parent='Hierarchy')
        cls.register_parser(name='MediumChunk', transform='get_deeper_nodes', depth=1, parent='Hierarchy')
        cls.register_parser(name='FineChunk', transform='get_deeper_nodes', depth=2, parent='Hierarchy')
        cls.register_parser(name='SentenceDivider', transform='SentenceSplitter', chunk_size=1024, chunk_overlap=20)
        cls.register_parser(name='TokenDivider', transform='TokenTextSplitter',
                            chunk_size=1024, chunk_overlap=20, separator=" ")
        cls.register_parser(name='HtmlExtractor', transform='HTMLNodeParser', tags=["p", "h1"])
        cls.register_parser(name='JsonExtractor', transform='JSONNodeParser')
        cls.register_parser(name='MarkDownExtractor', transform='MarkdownNodeParser')

    def __new__(cls, *args, **kw):
        lazyllm.call_once(register_flag, cls.register)
        return super().__new__(cls, *args, **kw)

    def __init__(self, embed, doc_path=None, doc_files=None, doc_name='LazyDocument'):
        super().__init__()
        self.doc_path, self.doc_files, self._embed = doc_path, doc_files, embed
        from ..rag.component.sent_embed import LLamaIndexEmbeddingWrapper
        self.embed = LLamaIndexEmbeddingWrapper(embed)

        self.docs = None
        self.nodes_dict = copy.deepcopy(DocImpl.parser_rule_dict)
        self.store = LazyStore(doc_name)

    def load_files(self, doc_path=None, doc_files=None):
        if not doc_files and not doc_path:
            from llama_index.core.schema import TextNode
            # empty will make error when create bm25. add "empty doc" to avoid this.
            return [TextNode(text="empty doc", metadata={"file_name": "empty_doc"})]

        return llama_index.core.SimpleDirectoryReader(input_dir=doc_path, input_files=doc_files).load_data()

    @classmethod
    def register_similarity(cls, func):
        cls.registered_similarity[func.__name__] = func
        return func

    @staticmethod
    def get_llamaindex_transform(name):
        assert isinstance(name, str), f'Transform name must be str, bug got {type(name)}'
        module = importlib.import_module('llama_index.core.node_parser')
        clazz = getattr(module, name, None)
        if clazz is None:
            raise NotImplementedError(f"Cannot find {name} in llamaindex.core.node_parser")
        return clazz

    @staticmethod
    def _register_parser(cls, name, transform, parent=None, **kwargs):
        assert name not in cls, f'Duplicate rule: {name}'
        cls[name] = dict(parser=transform if callable(transform) else DocImpl.get_llamaindex_transform(transform),
                         parser_kw=kwargs, parent_name=parent, nodes=None, retrievers_similarity={}, retrievers={})

    @classmethod
    def register_parser(cls, name, transform, parent=None, **kw):
        DocImpl._register_parser(cls.parser_rule_dict, name, transform, parent, **kw)

    def add_parse(self, name, transform, parent=None, **kw):
        DocImpl._register_parser(self.nodes_dict, name, transform, parent, **kw)
        return self

    def add_similarity(self, signature, similarity, similarity_kw, parser):
        assert parser in self.nodes_dict
        self.nodes_dict[parser]['retrievers_similarity'][signature] = {
            'similarity': similarity,
            'similarity_kw': similarity_kw,
        }

    def generate_signature(self, similarity, similarity_kw, parser):
        sorted_kw = sorted(similarity_kw.items())
        kw_str = ', '.join(f'{k}={v}' for k, v in sorted_kw)
        signature = f'{similarity}({kw_str})'
        hashed_signature = hashlib.sha256(signature.encode()).hexdigest()
        self.add_similarity(hashed_signature, similarity, similarity_kw, parser)
        return hashed_signature

    def _get_node(self, name):
        self.store.lazy_init()
        node = self.nodes_dict.get(name)
        if node is None:
            raise ValueError(f"Parser '{name}' does not exist. "
                             "Please check the parser name or add a new one through 'add_parse'.")
        if node['nodes'] is not None:
            return node
        if self.store.has_nodes(name):
            node['nodes'] = self.store.get_nodes(name)
        else:
            if node['parent_name']:
                parent_node = self._get_node(node['parent_name'])
                node['nodes'] = node['parser'](parent_node['nodes'], **node['parser_kw'])
            else:
                if self.docs is None:
                    self.docs = self.load_files(doc_path=self.doc_path, doc_files=self.doc_files)
                parser = node['parser'].from_defaults(**node['parser_kw'])
                node['nodes'] = parser.get_nodes_from_documents(self.docs)
            self.store.add_nodes(name, node['nodes'])
        return node

    def get_retriever(self, name, signature):
        node = self._get_node(name)
        if signature in node['retrievers']:
            return node['retrievers'][signature]
        if signature in node['retrievers_similarity']:
            func_info = node['retrievers_similarity'][signature]
            assert func_info['similarity'] in self.registered_similarity, (
                f"Unable to find retriever similarity {func_info['similarity']}, "
                "please check the similarity name or register a new one.")
            retriever = self.registered_similarity[func_info['similarity']](
                name, self.nodes_dict[name], self.embed, func_info['similarity_kw'], self.store)
            node['retrievers'][signature] = retriever
            return retriever
        else:
            raise ValueError(f"Func '{signature}' donse not exist.")

    def _query_with_sig(self, string, signature, parser):
        if type(string) is LazyLlmRequest:
            string = string.input
        retriever = self.get_retriever(parser, signature)
        if not isinstance(string, llama_index.core.schema.QueryBundle):
            string = llama_index.core.schema.QueryBundle(string)
        res = retriever.retrieve(string)
        return res

    def query(self, string, similarity, parser, **kw):
        sig = self.generate_signature(similarity, kw, parser)
        return self._query_with_sig(string, sig, parser)

    def _get_nodes_sort_list(self):
        nodes_sort_list = []
        for parser_name, node in self.nodes_dict.items():
            if not node["nodes"] or len(node["nodes"]) == 0:
                continue

            if node['parent_name'] is None:
                nodes_sort_list.append(parser_name)
            else:
                new_list = [parser_name]
                parent_name = node['parent_name']
                while parent_name is not None:
                    new_list.append(parent_name)
                    parent_name = self.nodes_dict[parent_name]['parent_name']

                for parser_name in reversed(new_list):
                    if parser_name not in nodes_sort_list:
                        nodes_sort_list.append(parser_name)
        return nodes_sort_list

    def add_files(self, input_files):
        if len(input_files) == 0: return

        for parser_name in self._get_nodes_sort_list():
            node = self.nodes_dict[parser_name]

            if node['parent_name']:
                parent_node = self.nodes_dict[node['parent_name']]
                add_nodes = node['parser'](parent_node['nodes'], **node['parser_kw'])
            else:
                parser = node['parser'].from_defaults(**node['parser_kw'])
                docs = self.load_files(doc_files=input_files)
                add_nodes = parser.get_nodes_from_documents(docs)

            self.store.add_nodes(parser_name, add_nodes)
            self.store.add_index_nodes(nodes_name=parser_name, nodes=add_nodes, embed_model=self.embed)

            node["nodes"] = node["nodes"] + add_nodes
            node["retrievers"] = {}
            for signature in node["retrievers_similarity"].keys():
                self.get_retriever(parser_name, signature)

    def delete_files(self, input_files):
        if len(input_files) == 0: return

        for parser_name in self._get_nodes_sort_list():
            node = self.nodes_dict[parser_name]

            delete_nodes = [
                it_node for it_node in node['nodes']
                if "file_name" in it_node.metadata and it_node.metadata["file_name"] in input_files
            ]

            if len(delete_nodes) == 0: continue

            self.store.del_nodes(parser_name, delete_nodes)
            self.store.del_index_nodes(parser_name, delete_nodes)
            del_node_ids = set([node.node_id for node in delete_nodes])
            node["nodes"] = [it_node for it_node in node["nodes"] if it_node.node_id not in del_node_ids]
            node["retrievers"] = {}
            for signature in node["retrievers_similarity"].keys():
                self.get_retriever(parser_name, signature)

@DocImpl.register_similarity
def defatult(name, nodes, embed, func_kw, store):
    index = store.get_index(
        nodes_name=name,
        nodes=nodes['nodes'],
        embed_model=embed,
    )
    return index.as_retriever(**func_kw)

@DocImpl.register_similarity
def chinese_bm25(name, nodes, embed, func_kw, store):
    from ..rag.component.bm25_retriever import ChineseBM25Retriever
    return ChineseBM25Retriever.from_defaults(nodes=nodes['nodes'], **func_kw)

@DocImpl.register_similarity
def bm25(name, nodes, embed, func_kw, store):
    from llama_index.retrievers.bm25 import BM25Retriever
    return BM25Retriever.from_defaults(nodes=nodes['nodes'], **func_kw)

class Retriever(ModuleBase):
    __enable_request__ = False

    def __init__(self, doc, parser, similarity='defatult', index='vector', **kw):
        super().__init__()
        self.doc = doc
        self.similarity = similarity
        self.similarity_kw = kw
        self.parser = parser
        self.index = index
        self.signature = self.doc.generate_signature(self.similarity, self.similarity_kw, self.parser)

    def forward(self, str):
        return self.doc._query_with_sig(str, self.signature, self.parser)


class Rerank(ModuleBase):
    registered_rerank = dict()

    def __init__(self, types='Reranker', **kwargs):
        super().__init__()
        self.type = types
        self.kw = kwargs
        self.kernel = None

    def forward(self, *inputs):
        if not self.kernel:
            self.kernel = self.get_rerank(self.type)
        if len(inputs) == 1:
            return self.kernel.postprocess_nodes(*inputs)
        elif len(inputs) == 2:
            if type(inputs[0]) is str:
                return self.kernel.postprocess_nodes(inputs[1], query_str=inputs[0])
            else:
                return self.kernel.postprocess_nodes(inputs[0], query_str=inputs[1])
        else:
            raise RuntimeError("Inputs len should be 1 or 2.")

    @classmethod
    def register_rerank(cls, func):
        cls.registered_rerank[func.__name__] = func
        return func

    def get_rerank(self, rerank):
        if rerank in self.registered_rerank:
            return self.registered_rerank[rerank](**self.kw)
        else:
            module = importlib.import_module('llama_index.core.postprocessor')
            clazz = getattr(module, rerank, None)
            if clazz is None:
                raise ImportError(
                    f"Class '{rerank}' is not registered and cannot be found "
                    "in '{module.__name__}'. Please use 'register_rerank' to register.")
            return clazz(**self.kw)

@Rerank.register_rerank
def Reranker(model, source=lazyllm.config['model_source'], top_n=-1):
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from lazyllm.components.utils.downloader import ModelDownloader
    model = ModelDownloader(source).download(model)
    return SentenceTransformerRerank(model=model, top_n=top_n)

@Rerank.register_rerank
def SimilarityFilter(threshold=0.003):
    from llama_index.core.postprocessor import SimilarityPostprocessor
    return SimilarityPostprocessor(similarity_cutoff=threshold)

@Rerank.register_rerank
def KeywordFilter(required_keys, exclude_keys):
    from llama_index.core.postprocessor import KeywordNodePostprocessor
    return KeywordNodePostprocessor(required_keywords=required_keys, exclude_keywords=exclude_keys)


lazyllm.config.add('rag_store', str, 'None', 'RAG_STORE').add('redis_url', str, 'None', 'REDIS_URL')


class LazyStore(object):

    def __new__(cls, *args, **kwargs):
        if cls == LazyStore:
            subclasses = {subclass.__name__.lower(): subclass for subclass in cls.__subclasses__()}
            name = lazyllm.config['rag_store'].lower() + 'store'
            if name in subclasses:
                return super().__new__(subclasses[name])
        return super().__new__(cls)

    def __init__(self, docs_name):
        self.docs_name = docs_name
        self.node_record = dict()
        self.index_dict = dict()

    def add_nodes(self, nodes_name, nodes): pass

    def add_index_nodes(self, nodes_name, nodes, embed_model):
        self.get_index(nodes_name, nodes, embed_model).insert_nodes(nodes)

    def del_nodes(self, nodes_name, nodes): pass

    def del_index_nodes(self, nodes_name, nodes):
        if nodes_name in self.index_dict:
            del self.index_dict[nodes_name]

    def get_nodes(self, nodes_name): pass

    def has_nodes(self, nodes_name): return False

    def lazy_init(self): pass

    def get_index(self, nodes_name, nodes, embed_model, rebuild=False):
        from llama_index.core import VectorStoreIndex
        if not rebuild and nodes_name in self.index_dict:
            return self.index_dict[nodes_name]
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            show_progress=True
        )
        self.index_dict[nodes_name] = index
        return index

class RedisStore(LazyStore):

    def __init__(self, docs_name):
        super().__init__(docs_name)
        self.init_flag = lazyllm.once_flag()
        self.redis_handle_name = 'lazyllmrag'

    def lazy_init(self):
        from lazyllm.thirdparty import redis
        from redisvl.schema import IndexSchema
        self.redis_client = redis.from_url(lazyllm.config['redis_url'])
        self.doc_store = self.create_document_store()
        self.node_record = self.load_base_infor_from_redis(self.docs_name)
        self.default_schema = IndexSchema.from_dict(
            {
                "index": {"name": "lazyllm", "prefix": "doc"},
                "fields": [
                    {"type": "tag", "name": "id"},
                    {"type": "tag", "name": "doc_id"},
                    {"type": "text", "name": "text"},
                    {
                        "type": "vector",
                        "name": "vector",
                        "attrs": {
                            "dims": 1024,
                            "algorithm": "flat",
                            "distance_metric": "cosine",
                        },
                    },
                ],
            }
        )

    def add_nodes(self, nodes_name, nodes):
        lazyllm.call_once(self.init_flag, self.lazy_init)
        save_dict = {nodes_name: [x.id_ for x in nodes]}
        self.node_record.update(save_dict)
        self.doc_store.add_documents(nodes)
        if not self.redis_client.sismember(self.redis_handle_name, self.docs_name):
            self.redis_client.sadd(self.redis_handle_name, self.docs_name)
        save_dict[nodes_name] = json.dumps(save_dict[nodes_name])
        for key, value in save_dict.items():
            self.redis_client.hset(self.docs_name, key, value)

    def del_nodes(self, nodes_name, nodes):
        lazyllm.call_once(self.init_flag, self.lazy_init)
        del_node_ids = [node.id_ for node in nodes]
        self.node_record[nodes_name] = [x for x in self.node_record[nodes_name] if x not in del_node_ids]

        for _id in del_node_ids:
            self.doc_store.delete_document(_id)
            self.redis_client.hdel(self.docs_name, _id)

        with self.redis_client.pipeline(transaction=False) as pipe:
            for _id in del_node_ids:
                pipe.delete(_id)
            pipe.execute()

    def del_index_nodes(self, nodes_name, nodes):
        if nodes_name not in self.index_dict:
            return

        vector_store = self.build_vector_store_context(nodes_name)
        with vector_store._index.client.pipeline(transaction=False) as pipe:
            for doc in nodes:
                pipe.delete(f"{self.docs_name}:{doc.node_id}")
            pipe.execute()

    def has_nodes(self, nodes_name):
        lazyllm.call_once(self.init_flag, self.lazy_init)
        return nodes_name in self.node_record

    def get_nodes(self, nodes_name):
        assert self.has_nodes(nodes_name)
        return [self.get_doc(node_id) for node_id in self.node_record[nodes_name]]

    def create_document_store(self):
        from llama_index.storage.docstore.redis import RedisDocumentStore
        return RedisDocumentStore.from_redis_client(
            redis_client=self.redis_client, namespace='lazyllm_rag'
        )

    def get_doc(self, doc_id):
        return self.doc_store.get_document(doc_id)

    def load_base_infor_from_redis(self, docs_name):
        if not self.redis_client.sismember(self.redis_handle_name, docs_name):
            return dict()
        retrieved_dict = self.redis_client.hgetall(docs_name)
        return {k.decode('utf-8'): json.loads(v.decode('utf-8')) for k, v in retrieved_dict.items()}

    @property
    def index_record(self):
        assert self.init_flag, "Redis client not initialized."
        return set([x.decode("utf-8") for x in self.redis_client.execute_command("FT._LIST")])

    def index_exists(self, nodes_name):
        return self.docs_name + '_' + nodes_name in self.index_record

    def build_vector_store_context(self, nodes_name):
        assert self.init_flag, "RedisStore not initialized."
        schema = self.default_schema.copy()
        schema.index.name = self.docs_name + '_' + nodes_name
        schema.index.prefix = self.docs_name

        from llama_index.vector_stores.redis import RedisVectorStore
        vector_store = RedisVectorStore(
            schema=schema,
            redis_client=self.redis_client,
            overwrite=False
        )
        return vector_store

    def get_index(self, nodes_name, nodes, embed_model, rebuild=False):
        assert self.init_flag, "RedisStore not initialized."
        from llama_index.core import VectorStoreIndex
        from llama_index.core import StorageContext
        if not rebuild and nodes_name in self.index_dict:
            return self.index_dict[nodes_name]
        if self.index_exists(nodes_name):
            nodes = []
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            storage_context=StorageContext.from_defaults(vector_store=self.build_vector_store_context(nodes_name))
        )
        self.index_dict[nodes_name] = index
        return index
