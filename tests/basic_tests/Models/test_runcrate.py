import lazyllm

from lazyllm.module.llms.onlinemodule.supplier.runcrate import RuncrateChat, RuncrateEmbed


def test_runcrate_chat_defaults_and_registration():
    module = RuncrateChat(api_key='rc_test', stream=False)

    assert module.series == 'runcrate'
    assert module._base_url == 'https://api.runcrate.ai/v1/'
    assert module._model_name == 'deepseek-ai/DeepSeek-V4.1-Flash'
    assert lazyllm.online.chat.runcrate is RuncrateChat


def test_runcrate_chat_accepts_catalog_model_ids():
    module = lazyllm.OnlineModule(
        model='zai-org/GLM-5.3',
        source='runcrate',
        url='https://api.runcrate.ai/v1/',
        api_key='rc_test',
    )

    assert isinstance(module, RuncrateChat)
    assert module._model_name == 'zai-org/GLM-5.3'


def test_runcrate_embedding_registration():
    assert lazyllm.online.embed.runcrate is RuncrateEmbed

    embedding = lazyllm.OnlineEmbeddingModule(
        model='Qwen/Qwen3-Embedding-0.6B',
        source='runcrate',
        url='https://api.runcrate.ai/v1/',
        api_key='rc_test',
    )
    assert embedding._embed_url == 'https://api.runcrate.ai/v1/embeddings'
    assert embedding._embed_model_name == 'Qwen/Qwen3-Embedding-0.6B'
