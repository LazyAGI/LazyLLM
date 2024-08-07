import lazyllm

# Note that if you cannot access the microphone, you need to enter the
# browser: chrome://flags/#unsafely-treat-insecure-origin-as-secure,
# fill in the access address URL, and agree to enable the microphone.

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'SenseVoiceSmall'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'SenseVoiceSmall') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/SenseVoiceSmall/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/SenseVoiceSmall`

chat = lazyllm.TrainableModule('SenseVoiceSmall')

if __name__ == '__main__':
    # Note that audio is enabled here
    lazyllm.WebModule(chat, port=8847, audio=True).start().wait()
