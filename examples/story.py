# -*- coding: utf-8 -*-

import lazyllm
from lazyllm import pipeline, warp, bind
from lazyllm.components.formatter import JsonFormatter

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'internlm2-chat-7b'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'internlm2-chat-7b') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/internlm2-chat-7b/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/internlm2-chat-7b`

toc_prompt = '''
You are a writing assistant. Generate a structured outline based on the user’s given topic. Your task is to understand the user's input, generate outline and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.

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
'''  # noqa: E501

completion_prompt = '''
You are now an intelligent assistant. Your task is to receive a dictionary containing `title` and `describe`, and expand the writing according to the guidance in `describe`.

Input example:
{
    "title": "# Level 1 Title",
    "describe": "This is the description for writing."
}

Output(Do not repeat "title"):
This is the expanded content for writing.
Receive as follows:

'''  # noqa: E501

writer_prompt = {'system': completion_prompt, 'user': '{"title": {title}, "describe": {describe}}'}

with pipeline() as ppl:
    # TODO: Each model can be configured with its own inference framework priority.
    ppl.outline_writer = lazyllm.TrainableModule('Qwen2.5-32B-Instruct').deploy_method(
        lazyllm.deploy.vllm).formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: '\n'.join([f'{o["title"]}\n{s}' for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))  # noqa: E501

if __name__ == '__main__':
    lazyllm.WebModule(ppl, port=range(23467, 24000)).start().wait()
