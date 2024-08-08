from lazyllm import TrainableModule, WebModule, deploy, pipeline, switch, _0

## 写提示词:
chatflow_intent_list = [ "聊天", "语音识别", "图片问答", "画图", "生成音乐", "文字转语音"]
agent_prompt = f"""
现在你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别。\n你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译。"intent_list"为所有意图名列表。\n
如果输入中带有attachments，根据attachments的后缀类型以最高优先级确定意图：如果是图像后缀如.jpg、.png等，则输出：图片问答；如果是音频后缀如.mp3、.wav等，则输出：语音识别。
## intent_list:\n{chatflow_intent_list}\n\n## 示例\nUser: 你好啊\nAssistant:  聊天\n
"""
painter_prompt = f'现在你是一位绘图提示词大师，能够将用户输入的任意中文内容转换成英文绘图提示词，在本任务中你需要将任意输入内容转换成英文绘图提示词，并且你可以丰富和扩充提示词内容。'
musician_prompt = f'现在你是一位作曲提示词大师，能够将用户输入的任意中文内容转换成英文作曲提示词，在本任务中你需要将任意输入内容转换成英文作曲提示词，并且你可以丰富和扩充提示词内容。'
## 大语言模型:
base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
chat = base.share().prompt()
## 组装应用:
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
## 启动应用：
if __name__ == '__main__':
    WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()
