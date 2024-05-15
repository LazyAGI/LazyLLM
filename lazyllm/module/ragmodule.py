import hashlib

from .module import ModuleBase
from ..common.common import LazyLlmRequest
from lazyllm.thirdparty import llama_index


class Document(ModuleBase):

    registered_retriever = dict()

    def __init__(self, doc_path, embed):
        super().__init__()
        self.doc_path, self._embed = doc_path, embed
        from ..rag.component.sent_embed import LLamaIndexEmbeddingWrapper
        self.embed = LLamaIndexEmbeddingWrapper(embed)

        self.docs = None
        self.nodes_dict = dict()
        self.default_retriever = None

    def load_files(self, doc_path):
        return llama_index.core.SimpleDirectoryReader(doc_path).load_data()

    @classmethod
    def register_retriever(cls, func):
        cls.registered_retriever[func.__name__] = func
        return func

    def add_parse(self, name, parser, parent=None, **kw):
        assert name not in self.nodes_dict
        self.nodes_dict[name] = {
            'parser': parser,
            'parser_kw': kw,
            'parent_name': parent,
            'nodes': None,
            'retrievers_algo':{},
            'retrievers':{},
        }
        return self

    def add_algo(self, signature, algo, algo_kw, parser):
        assert parser in self.nodes_dict
        self.nodes_dict[parser]['retrievers_algo'][signature] = {
            'algo': algo,
            'algo_kw': algo_kw,
        }

    def generate_signature(self, algo, algo_kw, parser):
        sorted_kw = sorted(algo_kw.items())
        kw_str = ', '.join(f'{k}={v}' for k, v in sorted_kw)
        signature = f'{algo}({kw_str})'
        hashed_signature = hashlib.sha256(signature.encode()).hexdigest()
        self.add_algo(hashed_signature, algo, algo_kw, parser)
        return hashed_signature
    

    def forward(self, string):
        if not self.default_retriever:
            from llama_index.core.node_parser import SentenceSplitter
            self.add_parse(name='lazy_default_retriever', parser=SentenceSplitter)
            self.default_retriever = Retriever(self, algo='vector',
                                               parser='lazy_default_retriever', similarity_top_k=3)
        return self.default_retriever.forward(string)

    def _get_node(self, name):
        node = self.nodes_dict.get(name)
        if node is None:
            raise ValueError(f"Parser '{name}' does not exist. "
                             "Please check the parser name or add a new one through 'add_parse'.")
        if node['nodes'] is not None:
            return node
        if node['parent_name']:
            parent_node = self._get_node(node['parent_name'])
            node['nodes'] = node['parser'](parent_node['nodes'], **node['parser_kw'])
        else:
            if self.docs is None:
                self.docs = self.load_files(self.doc_path)
            parser = node['parser'].from_defaults(**node['parser_kw'])
            node['nodes'] = parser.get_nodes_from_documents(self.docs)
        return node

    def get_retriever(self, name, signature):
        node = self._get_node(name)
        if signature in node['retrievers']:
            return node['retrievers'][signature]
        if signature in node['retrievers_algo']:
            func_info = node['retrievers_algo'][signature]
            assert func_info['algo'] in self.registered_retriever,\
                (f"Unable to find retriever algorithm {func_info['algo']}, "
                 "please check the algorithm name or register a new one.")
            retriever = self.registered_retriever[func_info['algo']](self.nodes_dict[name],
                                                                     self.embed, func_info['algo_kw'])
            node['retrievers'][signature] = retriever
            return retriever
        else:
            raise ValueError(f"Func '{signature}' donse not exist.")
 
    def _query_with_sig(self, string, signature, parser):
        if type(string) == LazyLlmRequest:
            string = string.input
        retriever = self.get_retriever(parser, signature)
        if not isinstance(string, llama_index.core.schema.QueryBundle):
            string = llama_index.core.schema.QueryBundle(string)
        res = retriever.retrieve(string)
        return res
    
    def query(self, string, algo, parser, **kw):
        sig = self.doc.generate_signature(algo, kw, parser)
        return self._query_with_sig(string, sig, parser)

@Document.register_retriever
def vector(nodes, embed, func_kw):
    index = llama_index.core.VectorStoreIndex(
        nodes = nodes['nodes'],
        embed_model = embed,
    )
    return index.as_retriever(**func_kw)

@Document.register_retriever
def bm25(nodes, embed, func_kw):
    from ..rag.component.bm25_retriever import ChineseBM25Retriever
    return ChineseBM25Retriever.from_defaults(
            nodes=nodes['nodes'],
            **func_kw
        )

class Retriever(ModuleBase):
    __enable_request__ = False

    def __init__(self, doc, algo, parser, **kw):
        super().__init__()
        self.doc = doc
        self.algo = algo
        self.algo_kw = kw
        self.parser = parser
        self.signature = self.doc.generate_signature(self.algo, self.algo_kw, self.parser)

    def forward(self, str):
        return self.doc._query_with_sig(str, self.signature, self.parser)
