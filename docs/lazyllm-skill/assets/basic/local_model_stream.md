# 本地模型与流式输出（本地参考）

本文档为 skill 本地摘要，完整内容见项目 `docs/zh/Best Practice/LocalModel.md` 或 `docs/en/Best Practice/LocalModel.md`。

## 启用流式输出

```python
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
```

## 使用 StreamCallHelper 迭代输出

```python
model = lazyllm.StreamCallHelper(model)
for msg in model('hello'):
    print(msg)
```

当模型在 Flow 中时，应包装最外层 Flow：

```python
model = lazyllm.TrainableModule('qwen2-1.5b', stream=True)
ppl = lazyllm.pipeline(model)
ppl = lazyllm.StreamCallHelper(ppl)
for msg in ppl('hello'):
    print(msg)
```

## 流式输出配置

可传入字典配置前缀、后缀与颜色：

```python
stream_config = {
    'color': 'green',
    'prefix': 'AI: ',
    'prefix_color': 'blue',
    'suffix': 'End\n',
    'suffix_color': 'red'
}
model = lazyllm.TrainableModule('qwen2-1.5b', stream=stream_config)
```

在线模型使用 `OnlineModule(..., stream=True)` 即可启用流式。
