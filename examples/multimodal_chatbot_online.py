import lazyllm

# Before running, set the environment variable:
#
# 1. `export LAZYLLM_GLM_API_KEY=xxxx`: the API key of Zhipu AI, you need to set `source="glm"` and
#     `model="glm-4v-flash"`. You can apply for the API key at https://open.bigmodel.cn/
#     Also supports other API keys:
#       - LAZYLLM_OPENAI_API_KEY: the API key of OpenAI, set `source="openai"` and `model="gpt-4o-mini"`.
#           You can apply for the API key at https://openai.com/index/openai-api/
#       - LAZYLLM_KIMI_API_KEY: the API key of Moonshot AI, set `source="kimi"` and
#           `model="moonshot-v1-8k-vision-preview"`.
#           You can apply for the API key at https://platform.moonshot.cn/console
#       - LAZYLLM_QWEN_API_KEY: the API key of Alibaba Cloud, set `source="qwen"` and `model="qwenvl-max"`.
#           You can apply for the API key at https://home.console.aliyun.com/
#       - LAZYLLM_SENSENOVA_API_KEY: the API key of SenseTime, set `source="sensenova"` and `model="SenseChat-Vision"`.
#                                  You also have to set LAZYLLM_SENSENOVA_SECRET_KEY` togather.
#           You can apply for the API key at https://platform.sensenova.cn/home
#     * `source` needs to be specified for multiple API keys, but it does not need to be set for a single API key.

chat = lazyllm.OnlineChatModule(source="sensenova", model="SenseNova-V6-5-Pro")

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=range(23466, 23470), files_target=chat).start().wait()
