from lazyllm.common.registry import LazyLLMRegisterMetaClass, LazyDict
from lazyllm.components.utils.downloader.model_downloader import LLMType
from .onlineChatModuleBase import OnlineChatModuleBase
from .onlineEmbeddingModuleBase import (
    OnlineEmbeddingModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineMultimodalEmbedModuleBase, LazyLLMOnlineRerankModuleBase
)
from .onlineMultiModalBase import (
    OnlineMultiModalBase, LazyLLMOnlineSTTModuleBase, LazyLLMOnlineTTSModuleBase,
    LazyLLMOnlineText2ImageModuleBase, LazyLLMOnlineImageEditingModuleBase
)


def _ensure_group(parent: LazyDict, name: str):
    if name in parent:
        grp = parent[name]
        assert isinstance(grp, LazyDict), (
            f'Group key conflict: "{name}" exists and is not a LazyDict'
        )
        return grp
    grp = LazyDict(name)
    parent[name] = grp
    return grp

online = LazyLLMRegisterMetaClass.all_clses['online']

embedding_group = _ensure_group(online, LLMType.EMBED)
embedding_group['base'] = OnlineEmbeddingModuleBase

multimodal_group = _ensure_group(online, 'multimodal')
multimodal_group['base'] = OnlineMultiModalBase


__all__ = [
    'OnlineChatModuleBase',
    'OnlineEmbeddingModuleBase',
    'LazyLLMOnlineEmbedModuleBase',
    'LazyLLMOnlineMultimodalEmbedModuleBase',
    'LazyLLMOnlineRerankModuleBase',
    'OnlineMultiModalBase',
    'LazyLLMOnlineSTTModuleBase',
    'LazyLLMOnlineTTSModuleBase',
    'LazyLLMOnlineText2ImageModuleBase',
    'LazyLLMOnlineImageEditingModuleBase'
]
