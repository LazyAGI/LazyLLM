import lazyllm

# ChatTTS supports 30 seconds of voice. However, the stability of the voice roles is poor,
# so a random number seed is set here to lock the voice role.

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'ChatTTS'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'ChatTTS') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/ChatTTS/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/ChatTTS`

m = lazyllm.TrainableModule('ChatTTS')
m.name = "tts"

if __name__ == '__main__':
    lazyllm.WebModule(
        m,
        port=12498,
        components={
            m: [('spk_emb', 'Text', 12)]
        }
    ).start().wait()
