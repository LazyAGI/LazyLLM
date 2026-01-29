from lazyllm import TrainableModule, WebModule, deploy, pipeline, switch, _0, IntentClassifier

chatflow_intent_list = ["聊天", "语音识别", "图片问答", "画图", "生成音乐", "文字转语音"]
agent_prompt = f'''
现在你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别。\n
你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译。"intent_list"为所有意图名列表。\n
如果输入中带有attachments，根据attachments的后缀类型以最高优先级确定意图：如果是图像后缀如.jpg、.png等，则输出：图片问答；\n
如果是音频后缀如.mp3、.wav等，则输出：语音识别。\n
## intent_list:\n{chatflow_intent_list}\n\n
# ## 示例\nUser: 你好啊\nAssistant:  聊天\n
'''
painter_prompt = '现在你是一位绘图提示词大师，能够将用户输入的任意中文内容转换成英文绘图提示词，'\
                 '在本任务中你需要将任意输入内容转换成英文绘图提示词，并且你可以丰富和扩充提示词内容。'
musician_prompt = '现在你是一位作曲提示词大师，能够将用户输入的任意中文内容转换成英文作曲提示词，'\
                  '在本任务中你需要将任意输入内容转换成英文作曲提示词，并且你可以丰富和扩充提示词内容。'

base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
chat = base.share().prompt()
with pipeline() as ppl:
    ppl.cls = base
    ppl.cls_normalizer = lambda x: x if x in chatflow_intent_list else chatflow_intent_list[0]
    with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
        ppl.sw.case[chatflow_intent_list[0], chat]
        ppl.sw.case[chatflow_intent_list[1], TrainableModule('SenseVoiceSmall')]
        ppl.sw.case[chatflow_intent_list[2], TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)]  # noqa: E501
        ppl.sw.case[chatflow_intent_list[3], pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]  # noqa: E501
        ppl.sw.case[chatflow_intent_list[4], pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]  # noqa: E501
        ppl.sw.case[chatflow_intent_list[5], TrainableModule('ChatTTS')]
WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()

# 代码优化
# 在 LazyLLM 中可以将 IntentClassifier 作为一个上下文管理器来使用
# 它内置了 Prompt, 我们仅需要指定 LLM 模型即可，它的使用方法和 Switch 类似，指定每个 case 分支，
# 并设置其中的意图类别和对应调用的对象即可。完整代码实现如下：

base = TrainableModule('internlm2-chat-7b')
with IntentClassifier(base) as ic:
    ic.case['聊天', base]
    ic.case['语音识别', TrainableModule('SenseVoiceSmall')]
    ic.case['图片问答', TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)]
    ic.case['画图', pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
    ic.case['生成音乐', pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
    ic.case['文字转语音', TrainableModule('ChatTTS')]
WebModule(ic, history=[base], audio=True, port=8847).start().wait()
