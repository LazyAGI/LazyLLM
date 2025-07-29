# 语音对话agent

# 本项目展示了如何使用[LazyLLM](https://github.com/LazyAGI/LazyLLM)，实现一个支持语音输入与语音播报的语音助手系统，可通过麦克风接收用户语音指令、识别语音文本、调用大模型生成回答，并通过语音播报返回。

# !!! abstract "通过本节您将学习到以下内容"
# - 如何使用 `speech_recognition` 接收并识别麦克风语音。
# - 如何使用 `LazyLLM.OnlineChatModule` 调用大模型进行自然语言回答。
# - 如何使用 `pyttsx3` 将文本转为语音实现播报。


# 项目依赖
# 确保安装以下依赖：
# pip install lazyllm pyttsx3 speechrecognition

import speech_recognition as sr
import pyttsx3
import lazyllm


# 步骤详解
# Step 1: 初始化大模型与语音播报引擎

chat = lazyllm.OnlineChatModule()
engine = pyttsx3.init()

# chat: 使用 LazyLLM 提供的在线聊天模块调用大模型（默认使用 sensenova 接口）。
# engine: 初始化本地语音合成引擎 pyttsx3，用于将文本转换为语音。

# Step 2: 构建语音助手主逻辑
# 定义主函数 listen(chat)，通过麦克风持续监听用户语音、识别成文本、调用大模型进行回答，并通过语音播报返回结果。

def listen(chat):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)
        print("Okay, go!")
        while 1:
            text = ""
            print("listening now...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                print("Recognizing...")
                text = r.recognize_whisper(
                    audio,
                    model="medium.en",
                    show_dict=True,
                )["text"]
            except Exception as e:
                unrecognized_speech_text = (
                    f"Sorry, I didn't catch that. Exception was: {e}s"
                )
                text = unrecognized_speech_text
            print(text)

            response_text = chat(text)
            print(response_text)
            engine.say(response_text)
            engine.runAndWait()

# 示例运行结果

# 示例场景：

# 你说：
# “What is the capital of France？”

# 程序控制台输出：
# Calibrating...
# Okay, go!
# listening now...
# Recognizing...
# You said: What is the capital of France?
# The capital of France is Paris.

# 系统语音播报：
# “The capital of France is Paris.”
