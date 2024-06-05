import lazyllm
from lazyllm import ModuleBase, ServerModule

from .web import DocWebModule
from .doc_manager import DocManager
from .group_doc import DocGroupImpl


class Document(ModuleBase):
    def __init__(self, dataset_path: str, embed, create_ui: bool = True, launcher=None):
        super().__init__()
        self._create_ui = create_ui
        launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)

        if create_ui:
            self._impl = DocGroupImpl(dataset_path=dataset_path, embed=embed)
            doc_manager = DocManager(self._impl)
            self.doc_server = ServerModule(doc_manager, launcher=launcher)

            self.web = DocWebModule(doc_server=self.doc_server)
        else:
            self._impl = DocGroupImpl(dataset_path=dataset_path, embed=embed)

    def generate_signature(self, similarity, similarity_kw, parser):
        return self._impl.generate_signature(similarity, similarity_kw, parser)

    def _query_with_sig(self, string, signature, parser):
        if self._create_ui:
            return self.doc_server(string, parser=parser, signature=signature)
        else:
            return self._impl.query_with_sig(string=string, signature=signature, parser=parser)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Document', create_ui=self._create_ui)
