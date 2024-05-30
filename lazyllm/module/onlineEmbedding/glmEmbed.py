import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class GLMEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://open.bigmodel.cn/api/paas/v4/embeddings",
                 embed_model_name: str = "embedding-2"):
        super().__init__(embed_url, lazyllm.config["glm_api_key"], embed_model_name)
