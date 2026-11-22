from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat

# OpenAI shut these down on 2026-10-23 (https://platform.openai.com/docs/deprecations).
SHUT_DOWN = {'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'}


def test_openai_chat_default_model_is_served():
    chat = OpenAIChat(api_key='test-key')
    assert chat._model_name not in SHUT_DOWN
