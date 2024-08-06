import lazyllm
import os
from lazyllm import globals

# Note that if you cannot access the microphone, you need to enter the
# browser: chrome://flags/#unsafely-treat-insecure-origin-as-secure,
# fill in the access address URL, and agree to enable the microphone.

chat = lazyllm.TrainableModule('SenseVoiceSmall')

if __name__ == '__main__':
    # Note that audio is enabled here
    lazyllm.WebModule(chat, port=8847, audio=True).start().wait()