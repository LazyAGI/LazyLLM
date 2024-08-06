import lazyllm

# ChatTTS supports 30 seconds of voice. However, the stability of the voice roles is poor,
# so a random number seed is set here to lock the voice role.

chat = lazyllm.TrainableModule('ChatTTS')
m = lazyllm.ServerModule(chat) # This is redundant and can actually be removed. It only verifies the validity of parameter passing.
m.name = "tts"

if __name__ == '__main__':
    lazyllm.WebModule(
        m,
        port=12498,
        components={
            m: [('spk_emb', 'Text', 12)]
        }
    ).start().wait()