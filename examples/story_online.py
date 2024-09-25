# -*- coding: utf-8 -*-

import lazyllm
from lazyllm import pipeline, warp, bind
from lazyllm.components.formatter import JsonFormatter

# Before running, set the environment variable:
#
# 1. `export LAZYLLM_GLM_API_KEY=xxxx`: the API key of Zhipu AI, default model "glm-4", `source="glm"`.
#     You can apply for the API key at https://open.bigmodel.cn/
#     Also supports other API keys:
#       - LAZYLLM_OPENAI_API_KEY: the API key of OpenAI, default model "gpt-3.5-turbo", `source="openai"`.
#           You can apply for the API key at https://openai.com/index/openai-api/
#       - LAZYLLM_KIMI_API_KEY: the API key of Moonshot AI, default model "moonshot-v1-8k", `source="kimi"`.
#           You can apply for the API key at https://platform.moonshot.cn/console
#       - LAZYLLM_QWEN_API_KEY: the API key of Alibaba Cloud, default model "qwen-plus", `source="qwen"`.
#           You can apply for the API key at https://home.console.aliyun.com/
#       - LAZYLLM_SENSENOVA_API_KEY: the API key of SenseTime, default model "SenseChat-5", `source="sensenova"`.
#                                  You also have to set LAZYLLM_SENSENOVA_SECRET_KEY` togather.
#           You can apply for the API key at https://platform.sensenova.cn/home
#     * `source` needs to be specified for multiple API keys, but it does not need to be set for a single API key.

toc_prompt = """
You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.

Please generate the corresponding list of nested dictionaries based on the following user input:

Example output:
[
    {
        "title": "# Level 1 Title",
        "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
    },
    {
        "title": "## Level 2 Title",
        "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
    },
    {
        "title": "### Level 3 Title",
        "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
    }
]
User input is as follows:
"""  # noqa: E50E

completion_prompt = """
You are now an intelligent assistant. Your task is to receive a dictionary containing `title` and `describe`, and expand the writing according to the guidance in `describe`.

Input example:
{
    "title": "# Level 1 Title",
    "describe": "This is the description for writing."
}

Output(Do not repeat "title"):
This is the expanded content for writing.
Receive as follows:

"""  # noqa: E50E

writer_prompt = {"system": completion_prompt, "user": '{"title": {title}, "describe": {describe}}'}

with pipeline() as ppl:
    ppl.outline_writer = lazyllm.OnlineChatModule(stream=False).formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(lazyllm.OnlineChatModule(stream=False).prompt(writer_prompt))
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))  # noqa: E50E

if __name__ == '__main__':
    lazyllm.WebModule(ppl, port=range(23467, 24000)).start().wait()
