import weakref
from pathlib import Path
from lazyllm import ModuleBase
from lazyllm.tools.servers.graphrag.graphrag_server_module import GraphRagServerModule

from .document import Document


class GraphDocument(ModuleBase):
    def __init__(self, document: Document):
        super().__init__()
        self._kg_dir = str(Path(document._manager._dataset_path) / '.graphrag_kg')
        self._graphrag_server_module = GraphRagServerModule(kg_dir=self._kg_dir)
        self._graphrag_index_task_id = None
        self._document = document
        document._graph_document = weakref.ref(self)

    def start(self):
        self._graphrag_server_module.start()

    def stop(self):
        self._graphrag_server_module.stop()
        self._graphrag_index_task_id = None

    def init_graphrag_kg(self, regenerate_config: bool = True):
        m = self._graphrag_server_module
        kb_files = self._document._list_all_files_in_dataset()
        m.prepare_files(kb_files, regenerate_config=regenerate_config)

    def start_graphrag_index(self, override: bool = True) -> str:
        m = self._graphrag_server_module
        res = m.create_index(override=override)
        self._graphrag_index_task_id = res['task_id']
        return 'Success'

    def graphrag_index_status(self) -> dict:
        m = self._graphrag_server_module
        res = m.index_status(self._graphrag_index_task_id)
        return res

    def query(self, query: str) -> str:
        m = self._graphrag_server_module
        res = m.query(query)
        return res['answer']

    def __del__(self):
        self._graphrag_server_module.stop()
        self._document._graph_documents = None


class UrlGraphDocument(ModuleBase):
    def __init__(self, graphrag_url: str):
        super().__init__()
        self._graphrag_server_url = graphrag_url

    def forward(self, *args, **kw):
        return GraphRagServerModule.query_by_url(self._graphrag_server_url, *args, **kw)
