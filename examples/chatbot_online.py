import lazyllm

# Before running, just set one of the following environment variables: LAZYLLM_OPENAI_API_KEY,
# LAZYLLM_KIMI_API_KEY, LAZYLLM_GLM_API_KEY, LAZYLLM_QWEN_API_KEY, LAZYLLM_QWEN_API_KEY,
# LAZYLLM_SENSENOVA_API_KEY, and then `source` can be left unset.

chat = lazyllm.OnlineChatModule(source="glm")

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=23466).start().wait()
