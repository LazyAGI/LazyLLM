# Voice Dialogue Agent
# This project demonstrates how to use LazyLLM to build a voice assistant system that supports speech input and audio output. It captures voice input through a microphone, transcribes it into text, generates a response using a large language model, and speaks the result aloud.

# !!! abstract "In this section, you will learn how to:"
# Use speech_recognition to capture and recognize voice input from a microphone.
# Use LazyLLM.OnlineChatModule to invoke a large language model for natural language responses.
# Use pyttsx3 to convert text to speech for spoken output.

# Project Dependencies
## Ensure the following dependencies are installed:
```bash
pip install lazyllm pyttsx3 speechrecognition
```
```
import speech_recognition as sr
import pyttsx3
import lazyllm
```

# Step-by-Step Breakdown
## Step 1: Initialize the LLM and Text-to-Speech Engine

```python
chat = lazyllm.OnlineChatModule()
engine = pyttsx3.init()
```

**Function Description：**
- `chat`: Uses LazyLLM's online chat module (default sensenova API)
  - Supports switching different LLM backends
  - Automatically manages conversation context
- `engine`: Initializes local text-to-speech engine (pyttsx3)
  - Cross-platform speech output
  - Supports adjusting speech rate, volume and other parameters

## Step 2: 构建语音助手主逻辑

``` python
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
```

## Sample Output

#### Example Scenario:

**You say:**  
"What is the capital of France?"

**Console output:**
```
Calibrating...
Okay, go!
listening now...
Recognizing...
You said: What is the capital of France?
The capital of France is Paris.
```

**System speech output:**  
"The capital of France is Paris."