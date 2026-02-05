# 启用流式输出
# 在创建时启用流式输出
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)

# 流式输出的使用
model = lazyllm.StreamCallHelper(model)
for msg in model('hello'):
    print(msg)

# 借助一个 StreamCallHelper 包装的模型，即可对模型的调用结果进行迭代，来达到流式输出的目标。
# 当模型在一个flow中的时候，需要包装最外层的flow，而不是模型，例如：
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
ppl = lazyllm.pipeline(model)
ppl = lazyllm.StreamCallHelper(ppl)
for msg in ppl('hello'):
    print(msg)

# 流式输出配置
# 可以给流式输出做一些配置，来给流式输出的内容加一些前缀或者后缀，并实现“花花绿绿”的流式输出。具体的配置如下：
# 配置流式输出样式
stream_config = {
    'color': 'green',           # 输出颜色
    'prefix': 'AI: ',          # 前缀
    'prefix_color': 'blue',    # 前缀颜色
    'suffix': 'End\n',            # 后缀
    'suffix_color': 'red'      # 后缀颜色
}

model = lazyllm.TrainableModule('qwen2-1.5b', stream=stream_config)
