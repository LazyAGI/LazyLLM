# -*- coding: utf-8 -*-
# flake8: noqa: F501

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'internlm2-chat-7b'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'internlm2-chat-7b') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/internlm2-chat-7b/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/internlm2-chat-7b`

from lazyllm import TrainableModule, WebModule, deploy, pipeline, switch, _0

# Write prompt words:
chatflow_intent_list = ["Chat", "Speech Recognition", "Image QA", "Drawing", "Generate Music", "Text to Speech"]
agent_prompt = f"""
You are now an intent classification engine, responsible for analyzing user input text based on dialogue information and determining a unique intent category.\nOnly reply with the name of the intent, do not output any additional fields, and do not translate. "intent_list" is the list of all intent names.\n
If the input contains attachments, determine the intent based on the attachment file extension with the highest priority: if it is an image extension like .jpg, .png, etc., then output: Image QA; if it is an audio extension like .mp3, .wav, etc., then output: Speech Recognition.
## intent_list:\n{chatflow_intent_list}\n\n## Example\nUser: Hello\nAssistant: Chat
"""
painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'
# Large language model:
base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
chat = base.share().prompt()
# Assemble application:
with pipeline() as ppl:
    ppl.cls = base
    ppl.cls_normalizer = lambda x: x if x in chatflow_intent_list else chatflow_intent_list[0]
    with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
        ppl.sw.case[chatflow_intent_list[0], chat]
        ppl.sw.case[chatflow_intent_list[1], TrainableModule('SenseVoiceSmall')]
        ppl.sw.case[chatflow_intent_list[2], TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)]
        ppl.sw.case[chatflow_intent_list[3], pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
        ppl.sw.case[chatflow_intent_list[4], pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
        ppl.sw.case[chatflow_intent_list[5], TrainableModule('ChatTTS')]
# Start application:
if __name__ == '__main__':
    WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()
