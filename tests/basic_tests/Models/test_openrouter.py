import lazyllm

from lazyllm.module.llms.onlinemodule.supplier.openrouter import OpenRouterChat


def test_openrouter_chat_defaults_and_registration():
    module = OpenRouterChat(api_key='sk-or-test', stream=False)

    assert module.series == 'openrouter'
    assert module._base_url == 'https://openrouter.ai/api/v1/'
    assert module._model_name == 'openrouter/auto'
    assert lazyllm.online.chat.openrouter is OpenRouterChat


def test_openrouter_chat_accepts_catalog_model_ids():
    module = lazyllm.OnlineModule(
        model='openrouter/free',
        source='openrouter',
        url='https://openrouter.ai/api/v1/',
        api_key='sk-or-test',
    )

    assert isinstance(module, OpenRouterChat)
    assert module._model_name == 'openrouter/free'
