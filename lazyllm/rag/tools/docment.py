from lazyllm import ModuleBase, ServerModule, LazyLlmRequest
from lazyllm.launcher import EmptyLauncher

from .doc_web_module import DocWebModule
from .doc_manager import DocumentManager
from .doc_impl import DocumentImpl

# ModuleBase
class Document(ModuleBase):
    def __init__(self, dataset_path: str, embed, create_ui: bool = True, launcher=EmptyLauncher(sync=False)):
        super().__init__()
        self._create_ui = create_ui

        if create_ui:
            self.doc_impl = DocumentImpl(dataset_path=dataset_path, embed=embed)
            doc_manager = DocumentManager(self.doc_impl)
            self.doc_server = ServerModule(doc_manager, launcher=launcher)

            self.web = DocWebModule(doc_server=self.doc_server)
        else:
            self.doc_impl = DocumentImpl(dataset_path=dataset_path, embed=embed)

    def generate_signature(self, algo, algo_kw, parser):
        return self.doc_impl.sub_doc.generate_signature(algo, algo_kw, parser)

    def _query_with_sig(self, string, signature, parser):
        if self._create_ui:
            return self.doc_server(LazyLlmRequest(input={
                "string": string,
                "parser": parser,
                "signature": signature
            }))
        else:
            self.doc_impl.sub_doc._query_with_sig(string=string, signature=signature, parser=parser)
