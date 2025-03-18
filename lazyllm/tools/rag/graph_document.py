import os

from typing import Callable, Optional, Dict, Union, List
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher

from .doc_manager import DocManager
from .doc_impl import StorePlaceholder, EmbedPlaceholder
from .graph_rag import GraphDocImpl
from .doc_node import DocNode
from .store_base import EMBED_DEFAULT_KEY
from .utils import DocListManager
from .web import DocWebModule


class CallableDict(dict):
    def __call__(self, cls, *args, **kw):
        return self[cls](*args, **kw)


class GraphDocument(ModuleBase):
    class _Manager(ModuleBase):

        def __init__(
            self,
            dataset_path: str,
            llm: ModuleBase = None,
            embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
            manager: Union[bool, str] = False,
            server: bool = False,
            name: Optional[str] = None,
            launcher: Optional[Launcher] = None,
            store_conf: Optional[Dict] = None,
            reader_conf: Optional[Dict] = None,
        ):
            super().__init__()
            self._origin_path = dataset_path
            if not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            else:
                dataset_path = os.path.join(os.getcwd(), dataset_path)
            self._launcher: Launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = self._get_embeds(embed)
            assert llm is not None, "llm is required for GraphDocument"
            self._llm = llm
            self.name = name
            self._dlm = DocListManager(dataset_path, name, enable_path_monitoring=False if manager else True)

            self._kbs = CallableDict(
                {
                    DocListManager.DEFAULT_GROUP_NAME: GraphDocImpl(
                        llm=self._llm,
                        embed=self._embed,
                        group_name="",
                        reader_conf=reader_conf,
                        store_conf=store_conf,
                        document_module_id=self._module_id,
                        dlm=self._dlm,
                    )
                }
            )
            if manager:
                self._manager = ServerModule(DocManager(self._dlm))
            if manager == 'ui':
                self._docweb = DocWebModule(doc_server=self._manager)
            if server:
                self._kbs = ServerModule(self._kbs)

        @property
        def url(self):
            if hasattr(self, '_manager'):
                return self._manager._url
            return None

        @property
        @deprecated('GraphDocument.manager.url')
        def _url(self):
            return self.url

        @property
        def web_url(self):
            if hasattr(self, '_docweb'):
                return self._docweb.url
            return None

        def _get_embeds(self, embed):
            embeds = embed if isinstance(embed, dict) else {EMBED_DEFAULT_KEY: embed} if embed else {}
            for embed in embeds.values():
                if isinstance(embed, ModuleBase):
                    self._submodules.append(embed)
            return embeds

        def get_doc_by_kb_group(self, name):
            return self._kbs._impl._m[name] if isinstance(self._kbs, ServerModule) else self._kbs[name]

        def stop(self):
            if hasattr(self, '_docweb'):
                self._docweb.stop()
            self._launcher.cleanup()

        def __call__(self, *args, **kw):
            return self._kbs(*args, **kw)

    def __init__(
        self,
        dataset_path: str,
        llm: ModuleBase = None,
        embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
        create_ui: bool = False,
        manager: bool = False,
        server: bool = False,
        name: Optional[str] = None,
        launcher: Optional[Launcher] = None,
        store_conf: Optional[Dict] = None,
        reader_conf: Optional[Dict] = None,
    ):
        super().__init__()
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
        self._manager = GraphDocument._Manager(
            dataset_path, llm, embed, create_ui or manager, server, name, launcher, store_conf, reader_conf
        )
        self._curr_group = DocListManager.DEFAULT_GROUP_NAME

    @property
    @deprecated('GraphDocument._manager')
    def _impls(self):
        return self._manager

    @property
    def _impl(self):
        return self._manager.get_doc_by_kb_group(self._curr_group)

    @property
    def manager(self):
        return self._manager

    @DynamicDescriptor
    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        if isinstance(self, type):
            return GraphDocImpl.register_global_reader(pattern=pattern, func=func)
        else:
            self._impl.add_reader(pattern, func)

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
        return cls.add_reader(pattern, func)

    def get_store(self):
        return StorePlaceholder()

    def get_embed(self):
        return EmbedPlaceholder()

    def _forward(self, func_name: str, *args, **kw):
        return self._manager(self._curr_group, func_name, *args, **kw)

    def forward(self, *args, **kw) -> List[DocNode]:
        return self._forward('retrieve', *args, **kw)

    def __repr__(self):
        return lazyllm.make_repr(
            "Module",
            "GraphDocument",
            manager=hasattr(self._manager, '_manager'),
            server=isinstance(self._manager._kbs, ServerModule),
        )
