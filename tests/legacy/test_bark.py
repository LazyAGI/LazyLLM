import lazyllm

# Bark only supports 13-15 seconds of voice, and the Chinese voice is not very authentic.

chat = lazyllm.TrainableModule('bark')
m = lazyllm.ServerModule(chat) # This is redundant and can actually be removed. It only verifies the validity of parameter passing.
chat.name = "tts"

if __name__ == '__main__':
    lazyllm.WebModule(
        m,
        port=8847,
        components={
            chat:[('voice_preset', 'Dropdown', [
                "v2/zh_speaker_0",
                "v2/zh_speaker_1",
                "v2/zh_speaker_2",
                "v2/zh_speaker_3",
                "v2/zh_speaker_4",
                "v2/zh_speaker_5",
                "v2/zh_speaker_6",
                "v2/zh_speaker_7",
                "v2/zh_speaker_8",
                "v2/zh_speaker_9",
            ])],
        }
    ).start().wait()
