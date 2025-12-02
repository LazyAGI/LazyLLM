# Multimodal Chatbot

We will further enhance our chatbot! Based on the previous section [Painting Master](painting_master.md), we introduce more models to transform it into a multimodal robot! Let's get started!

!!! abstract "Through this section, you will learn the following key points of LazyLLM"

    - How to set different prompt words for the same model;
    - How to implement routing based on [Switch][lazyllm.flow.Switch] control flow, we will combine it with LLM to create a simple intent recognition robot;
    - How to specify the deployment framework for [TrainableModule][lazyllm.module.TrainableModule];
    - How to use [bind](../Best Practice/flow.md#use-bind) on the control flow to pass in parameters;
    - How to use [IntentClassifier][lazyllm.tools.IntentClassifier] for intent recognition.

## Design Concept

To expand the capabilities of our robot, we want it not only to paint but also to perform speech recognition, music composition, image-text Q&A, and have multimedia capabilities. We will introduce the following models:

- `ChatTTS`: For converting text to speech;
- `musicgen-small`: For generating music;
- `stable-diffusion-3-medium`: Continuing with the model from the previous section [Painting Master](painting_master.md), used for generating images;
- `internvl-chat-2b-v1-5`: For image-text Q&A;
- `SenseVoiceSmall`: For speech recognition;

We note that among the introduced models, there are ones for generating images and music, both of which require relatively high standards for prompt words. We will rely on an LLM to generate and translate prompt words, similar to what we did in the previous section [Painting Master](painting_master.md).

Furthermore, because we are introducing a large number of models, we need an intent recognition robot to forward user intents and route them to the corresponding model.

Considering the above, we proceed with the following design:

![Multimodal bot](../assets/3_multimodal-bot3.svg)

Here, the intent recognition robot is the core, routing user needs to one of the six functional robots.

It is worth noting that LazyLLM provides an intent recognition tool, see the [Code Optimization](#code_opt) section of this chapter for details.

## Code Implementation

Let's implement the above design concept based on LazyLLM.

### Designing Prompt

We need to design three prompt words for the intent recognition robot, painting generation robot, and music generation robot.

For the intent recognition robot, we design as follows:

```python
chatflow_intent_list = ["Chat", "Speech Recognition", "Image QA", "Drawing", "Generate Music", "Text to Speech"]
agent_prompt = f"""
You are now an intent classification engine, responsible for analyzing user input text based on dialogue information and determining a unique intent category.\nOnly reply with the name of the intent, do not output any additional fields, and do not translate. "intent_list" is the list of all intent names.\n
If the input contains attachments, determine the intent based on the attachment file extension with the highest priority: if it is an image extension like .jpg, .png, etc., then output: Image QA; if it is an audio extension like .mp3, .wav, etc., then output: Speech Recognition.
## intent_list:\n{chatflow_intent_list}\n\n## Example\nUser: Hello\nAssistant: Chat
"""
```

For the painting generation robot, we design as follows:

```python
painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
```

Similarly, there is the music generation robot:

```python
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'
```
### Setting up the Model

In this application design, we will use a total of four different LLMs, but in reality, the only difference between them lies in the prompts, so we use only one model `internlm2-chat-7b` as our shared robot, achieving different robots by setting different prompt templates.

Firstly, for our intent recognition robot, we will apply the newly designed prompt:

```python
base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
```

Then, for our chatbot, the prompt here is set to empty:

```python
chat = base.share().prompt()
```

[](){#use_share}

Here, we make use of the `share` functionality of `TrainableModule`. It allows setting different templates on top of the original robot. The object it returns is also a [TrainableModule][lazyllm.module.TrainableModule], but shares the model with the original.

Next are the painting and music generation robots, which also use the `share` functionality to achieve robots with different functions:

```python
painter = pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))
musician = pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))
```

Here, we use the non-context manager usage of `pipeline` 
(This section essentially follows the code from the previous section [Painting Master](painting_master.md), which uses the [Pipeline][lazyllm.flow.Pipeline] context manager.


For the remaining multimodal models, we can simply call them by specifying their names:

```python
stt = TrainableModule('SenseVoiceSmall')
vqa = TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)
tts = TrainableModule('ChatTTS')
```

It should be noted that `internvl-chat-2b-v1-5` is currently only supported in LMDeploy, so you need to explicitly specify the framework used for deployment, which is `.deploy_method(deploy.LMDeploy)` here.

### Assembling the Application

Let's assemble the models defined above:

```python
with pipeline() as ppl:
    ppl.cls = base
    ppl.cls_normalizer = lambda x: x if x in chatflow_intent_list else chatflow_intent_list[0]
    with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
        ppl.sw.case[chatflow_intent_list[0], chat]
        ppl.sw.case[chatflow_intent_list[1], TrainableModule('SenseVoiceSmall')]
        ppl.sw.case[chatflow_intent_list[2], TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)]
        ppl.sw.case[chatflow_intent_list[3], pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
        ppl.sw.case[chatflow_intent_list[4], pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
        ppl.sw.case[chatflow_intent_list[5], TrainableModule('ChatTTS')]
```

In the above code, we first instantiate a sequentially executing `ppl`. In this `ppl`, we first perform intent recognition and set `ppl.cls`.
Then, to ensure the robustness of intent recognition, we set an anonymous function `ppl.cls_normalizer` after intent recognition,
which maps the recognition results to the predefined list of attributes, ensuring that the recognition results are only within the predefined table.

```python
with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
```

Regarding this line of code:

- We first focus on `bind(_0, ppl.input)`, where `_0` is the first argument of the output from the previous step, which is an intent from the intent list. `ppl.input` is the user's input (corresponding to the red line in the design diagram). So this line of code sets two parameters for the `switch` control flow, the first being the intent and the second being the user's input. The usage introduction of bind can be found in detail at: [Parameter Binding](../Best Practice/flow.md#use-bind)

- Then `judge_on_full_input=False`, which allows the input to be divided into two parts: the first part as the condition for judgment, and the remaining part as the input for the branch. Otherwise, if `True`, the entire input will be used as both the condition for judgment and the branch input.

- Finally, we add the instantiated `switch` to `ppl`: `ppl.sw`. For more information, see: [Switch][lazyllm.flow.Switch].

The rest of the code follows a similar pattern, setting conditions and corresponding routing branches. Taking the following code as an example:

```python
ppl.sw.case[chatflow_intent_list[0], chat]
```

The first parameter of this code is the first element of the intent list, which is “chat”. This serves as the basis for the judgment condition. If the first parameter of the switch input is “chat”, then it will proceed to this branch chat. The input for chat is `ppl.input`.

### Starting the Application
Finally, we encapsulate the control flow `ppl` into a client and start the deployment (`start()`), and keep the client from closing after deployment (`wait()`).

```python
WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()
```

Here, since we need to use the microphone to capture sound, we have set `audio=True`.

## Full Code

<details>
<summary>Click to get import and prompt</summary>

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline, switch, _0

chatflow_intent_list = ["Chat", "Speech Recognition", "Image QA", "Drawing", "Generate Music", "Text to Speech"]
agent_prompt = f"""
You are now an intent classification engine, responsible for analyzing user input text based on dialogue information and determining a unique intent category.\nOnly reply with the name of the intent, do not output any additional fields, and do not translate. "intent_list" is the list of all intent names.\n
If the input contains attachments, determine the intent based on the attachment file extension with the highest priority: if it is an image extension like .jpg, .png, etc., then output: Image QA; if it is an audio extension like .mp3, .wav, etc., then output: Speech Recognition.
## intent_list:\n{chatflow_intent_list}\n\n## Example\nUser: Hello\nAssistant: Chat
"""
painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'
```
</details>

```python
base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
chat = base.share().prompt()
with pipeline() as ppl:
    ppl.cls = base
    ppl.cls_normalizer = lambda x: x if x in chatflow_intent_list else chatflow_intent_list[0]
    with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
        ppl.sw.case[chatflow_intent_list[0], chat]
        ppl.sw.case[chatflow_intent_list[1], TrainableModule('SenseVoiceSmall')]
        ppl.sw.case[chatflow_intent_list[2], TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)]
        ppl.sw.case[chatflow_intent_list[3], pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
        ppl.sw.case[chatflow_intent_list[4], pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
        ppl.sw.case[chatflow_intent_list[5], TrainableModule('ChatTTS')]
WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()
```

The effect is as follows:

![Multimodal Robot](../assets/3_multimodal-bot2.png)

## Code Optimization

[](){#code_opt}

In the above code, we have implemented a simple intent recognition using some basic modules of LazyLLM. However, this intent recognition code is somewhat redundant, complex to use, and lacks functions such as historical memory. Therefore, LazyLLM provides an intent recognition tool to simplify the implementation of intent recognition. We will design the modification as follows:

![Multimodal bot](../assets/3_multimodal-bot4.svg)

In LazyLLM, [IntentClassifier][lazyllm.tools.IntentClassifier] can be used as a context manager. It comes with a built-in Prompt, and we only need to specify the LLM model. Its usage is similar to [Switch][lazyllm.flow.Switch], where you specify each `case` branch and set the intent categories and the corresponding objects to be called. The complete code implementation is as follows:

<details>
<summary>click to look up prompts and imports</summary>

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline
from lazyllm.tools import IntentClassifier

painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'
```
</details>

```python
base = TrainableModule('internlm2-chat-7b')
with IntentClassifier(base) as ic:
    ic.case['Chat', base]
    ic.case['Speech Recognition', TrainableModule('SenseVoiceSmall')]
    ic.case['Image QA', TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)]
    ic.case['Drawing', pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
    ic.case['Generate Music', pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
    ic.case['Text to Speech', TrainableModule('ChatTTS')]
WebModule(ic, history=[base], audio=True, port=8847).start().wait()
```
