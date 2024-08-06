import lazyllm

# Bark only supports 13-15 seconds of voice, and the Chinese voice is not very authentic.

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'bark'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'bark') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/bark/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/bark`

m = lazyllm.TrainableModule('bark')
m.name = "tts"

if __name__ == '__main__':
    m.WebModule(
        m,
        port=8847,
        components={
            m: [('voice_preset', 'Dropdown', [
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
