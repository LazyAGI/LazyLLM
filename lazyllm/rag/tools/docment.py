import lazyllm
from lazyllm import ModuleBase, ServerModule
from lazyllm.launcher import EmptyLauncher

from .doc_web_module import DocWebModule
from .doc_manager import DocManager
from .doc_group_impl import DocGroupImpl

# ModuleBase
class Document(ModuleBase):
    def __init__(self, dataset_path: str, embed, create_ui: bool = True, launcher=EmptyLauncher(sync=False)):
        super().__init__()
        self._create_ui = create_ui

        if create_ui:
            self._impl = DocGroupImpl(dataset_path=dataset_path, embed=embed)
            doc_manager = DocManager(self._impl)
            self.doc_server = ServerModule(doc_manager, launcher=launcher)

            self.web = DocWebModule(doc_server=self.doc_server)
        else:
            self._impl = DocGroupImpl(dataset_path=dataset_path, embed=embed)

    def generate_signature(self, algo, algo_kw, parser):
        return self._impl.generate_signature(algo, algo_kw, parser)

    def _query_with_sig(self, string, signature, parser):
        if self._create_ui:
            return self.doc_server(string, parser=parser, signature=signature)
        else:
            return self._impl.query_with_sig(string=string, signature=signature, parser=parser)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Document', create_ui=self._create_ui)
