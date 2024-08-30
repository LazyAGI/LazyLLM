import lazyllm

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'internlm2-chat-7b'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'internlm2-chat-7b') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/internlm2-chat-7b/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/internlm2-chat-7b`

chat = lazyllm.TrainableModule('internlm2-chat-7b')

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=range(23466, 23470)).start().wait()
