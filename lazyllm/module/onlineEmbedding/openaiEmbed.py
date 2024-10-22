import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class OpenAIEmbedding(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = "https://api.openai.com/v1/embeddings",
                 embed_model_name: str = "text-embedding-ada-002",
                 api_key: str = None):
        super().__init__("OPENAI", embed_url, api_key or lazyllm.config['openai_api_key'], embed_model_name)
