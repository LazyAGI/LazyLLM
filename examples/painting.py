import lazyllm
from lazyllm import pipeline

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'internlm2-chat-7b'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'internlm2-chat-7b') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/internlm2-chat-7b/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/internlm2-chat-7b`

prompt = ('You are a drawing prompt word master who can convert any Chinese content entered by '
          'the user into English drawing prompt words. In this task, you need to convert any '
          'input content into English drawing prompt words, and you can enrich and expand the '
          'prompt word content.')

with pipeline() as ppl:
    ppl.llm = (lazyllm.TrainableModule('Qwen2.5-32B-Instruct')
               .deploy_method(lazyllm.deploy.vllm)
               .prompt(lazyllm.ChatPrompter(prompt)))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')

if __name__ == '__main__':
    lazyllm.WebModule(ppl, port=23466).start().wait()
