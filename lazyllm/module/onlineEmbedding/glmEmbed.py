import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class GLMEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://open.bigmodel.cn/api/paas/v4/embeddings",
                 embed_model_name: str = "embedding-2",
                 api_key: str = None):
        super().__init__("GLM", embed_url, api_key or lazyllm.config["glm_api_key"], embed_model_name)
