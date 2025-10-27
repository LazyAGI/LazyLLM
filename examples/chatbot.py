import lazyllm

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'Qwen2.5-32B-Instruct'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'Qwen2.5-32B-Instruct') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/Qwen2.5-32B-Instruct/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/Qwen2.5-32B-Instruct`

chat = lazyllm.TrainableModule('Qwen2.5-32B-Instruct').deploy_method(lazyllm.deploy.vllm)

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=range(23466, 23470)).start().wait()
