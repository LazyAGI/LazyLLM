"""
基础微调示例

使用方法:
1. 安装依赖: lazyllm install llama-factory
2. 准备训练数据
3. 运行: python basic_finetune.py
"""

import lazyllm
from lazyllm.tools import finetune
# 创建可训练模型
model = lazyllm.TrainableModule('internlm2-chat-7b')

# 准备训练数据
train_data = [
    {"query": "什么是机器学习？", "answer": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。"},
    {"query": "什么是深度学习？", "answer": "深度学习是机器学习的一种，使用多层神经网络来学习数据。"},
    {"query": "什么是自然语言处理？", "answer": "自然语言处理是人工智能的一个领域，研究如何让计算机理解和生成人类语言。"},
]

print("开始微调...")
print(f"训练数据量: {len(train_data)}")
print()

# 开始微调
model.finetune_method(
    finetune.llamafactory,
    learning_rate=1e-4,      # 学习率
    num_train_epochs=3,      # 训练轮数
    per_device_train_batch_size=4,  # 批次大小
    max_samples=1000,        # 最大样本数
    val_size=0.1)\
    .trainset(train_data)\
    .mode('finetune')

model.update()
print("\n微调完成！")

# 保存模型
output_dir = './finetuned_model'
model.save(output_dir)
print(f"模型已保存到: {output_dir}")

# 测试微调后的模型
print("\n测试微调后的模型...")
test_query = "什么是机器学习？"
result = model.forward(test_query)
print(f"问题: {test_query}")
print(f"回答: {result}")
