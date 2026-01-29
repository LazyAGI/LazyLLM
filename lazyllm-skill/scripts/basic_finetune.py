import lazyllm

# 使用自动微调方法
model = lazyllm.TrainableModule('qwen2-1.5b', target_path='/path/to/model') \
    .finetune_method(lazyllm.finetune.auto) \
    .trainset('/path/to/training/data') \
    .mode('finetune')

# 使用特定的微调方法
model = lazyllm.TrainableModule('qwen2-1.5b') \
    .finetune_method(lazyllm.finetune.llamafactory, learning_rate=1e-4, num_train_epochs=3) \
    .trainset('/path/to/training/data') \
    .mode('finetune')

# 执行微调
model.update()
