requirements = '''
collie-lm>=1.0.5
accelerate==0.20.3
appdirs
loralib==0.1.1
bitsandbytes
datasets
fire
peft==0.3.0
transformers>=4.32.0
sentencepiece==0.1.99
gradio
tokenizers==0.13.3
scipy
deepspeed==0.12.3
sentence_transformers
faiss-cpu
'''

collie_info ={
    'ability':{
        'tp': True,
        'pp': True,
        'zero': True,
    },
    'train_params':{
        'data_path': None,
        'batch_size': 64,
        'micro_batch_size': 4,
        'num_epochs': 3,
        'learning_rate': 5.e-4,
        'dp_size': 8,
        'pp_size': 1,
        'tp_size': 1,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'lora_target_modules': '[query_key_value, dense,dense_4h_to_h, dense_h_to_4h]',
        'modules_to_save': '[word_embeddings, output_layer]',
        'prompt_with_background': True,
    },
    'requrements': requirements
}