# 输出格式化
# 使用内置格式化器
# 使用 JSON 格式化器，可以通过 JsonFormatter 对输出中的json进行提取，并获取指定的元素。
import lazyllm
model = lazyllm.TrainableModule('qwen2-1.5b')\
    .formatter(lazyllm.formatter.JsonFormatter('[:][a]'))

# 使用自定义函数作为格式化器
def my_formatter(text):
    return f"处理后的结果: {text.strip()}"

model = lazyllm.TrainableModule('qwen2-1.5b').formatter(my_formatter)

# 使用链式格式化器
model = model.formatter(lazyllm.formatter.JsonFormatter() | lazyllm.formatter.StrFormatter())
