import lazyllm

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'Qwen3-30B-A3B-Instruct-2507'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'Qwen3-30B-A3B-Instruct-2507') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/Qwen3-30B-A3B-Instruct-2507/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/Qwen3-30B-A3B-Instruct-2507`

chat = lazyllm.TrainableModule('Qwen3-30B-A3B-Instruct-2507').deploy_method(lazyllm.deploy.vllm)

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=range(23466, 23470)).start().wait()
